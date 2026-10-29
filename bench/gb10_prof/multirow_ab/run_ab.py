#!/usr/bin/env python3
"""Linux A/B: does multirow drafting pay on GB10 at batch >= 3?

Infernix already implements choose_multirow_draft_length but gates it OFF
(INFERNIX_MULTIROW_DRAFTS=0.0) on a 5090-era measurement whose own code comment says
that call is wrong for GB10 (bandwidth-starved -> verifying draft columns is cheaper).
This mirrors dgpp's batched-MTP aggregate-throughput family.

Greedy (temperature 0): speculative verify must NOT change output, so the same prompt
text run in both arms must produce identical transcripts -> a bit-exactness gate on the
speculative path, independent of throughput.

Distinct-but-parallel prompts per burst avoid cross-request prefix-cache contamination;
identical prompt text across arms enables the transcript diff. Source of truth is the
serve request log (--request-log-jsonl), same schema as production log.json.
"""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

AB_DIR = os.path.dirname(os.path.abspath(__file__))
BIN = os.environ.get("AB_BIN", "/opt/etc/Infernix/build-native/apps/infernix-serve")
DIR = os.environ.get("AB_MODEL_DIR", "/opt/etc/models/Qwen3.8-Flash-Next-Nvidia-Dense8-Infernix")
MODEL = os.path.join(DIR, "Qwen3.8-Flash-Next-NVIDIA-NVFP4-Dense8-Infernix-00001-of-00003.infernix")
NGRAM = os.path.join(DIR, "Qwen3.8-Flash-Next-NVIDIA-NVFP4-Infernix.ngram")
HOST = os.environ.get("AB_HOST", "127.0.0.1")
PORT = int(os.environ.get("AB_PORT", "8091"))
NREQ = int(os.environ.get("AB_NREQ", "4"))
MAXTOK = int(os.environ.get("AB_MAXTOK", "192"))
REPEATS = int(os.environ.get("AB_REPEATS", "2"))
RESULTS = os.path.join(AB_DIR, "results.tsv")

# Distinct reasoning-style prompts: long greedy continuations, shared domain so all rows
# co-decode many steps, different content so they do not prefix-cache onto one another.
PROMPTS = [
    "Explain step by step how a transformer decodes one token at a time and why memory "
    "bandwidth, not arithmetic, usually limits single-stream inference on a modern integrated "
    "GPU. Cover the KV cache, why attention over a long context grows traffic, why reading the "
    "weights every step dominates small-batch decode, and what speculative decoding changes "
    "about the tradeoff between extra verification work and fewer sequential steps.",
    "Describe, in numbered steps, how Mixture-of-Experts routing works in a large language "
    "model: how the router scores experts, how top-k experts are chosen, how their outputs are "
    "combined, and why only a small fraction of expert weights is touched per token. Then "
    "explain why expert weights that are not used still cost memory capacity and why caching "
    "hot experts helps when the memory system is bandwidth-limited.",
    "Walk through how grouped-query attention differs from multi-head attention. Explain the "
    "role of query heads versus shared key and value heads, how this changes the size of the "
    "KV cache, and why a smaller KV cache matters for long-context decoding when memory "
    "bandwidth is scarce. End with a short list of the tradeoffs this design accepts.",
    "Explain the mechanics of layer normalization and residual connections in a deep "
    "transformer, then describe what hyper-connections add on top of ordinary residuals. "
    "Cover why widening the residual path can help very deep models train stably, and what "
    "extra memory traffic that widening introduces during token-by-token decoding.",
]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def port_open(port):
    s = socket.socket(); s.settimeout(1)
    try:
        s.connect((HOST, port)); return True
    except Exception:
        return False
    finally:
        s.close()


def model_ready():
    try:
        with urllib.request.urlopen("http://%s:%d/v1/models" % (HOST, PORT), timeout=5) as r:
            return len(json.loads(r.read().decode()).get("data", [])) > 0
    except Exception:
        return False


def gpu_used_mib():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15)
        return int(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def settle_gpu(timeout=120, threshold=6000):
    deadline = time.time() + timeout
    while time.time() < deadline:
        mib = gpu_used_mib()
        if mib is not None and mib < threshold:
            return
        log("waiting for GPU to release (%s MiB)..." % mib)
        time.sleep(3)


def start_serve(env_extra, serve_log, req_log):
    if os.path.exists(req_log):
        os.remove(req_log)
    env = dict(os.environ); env.update(env_extra)
    args = [BIN, MODEL, "--ngram-volume", NGRAM,
            "--host", HOST, "--port", str(PORT),
            "--max-concurrency", str(NREQ), "--max-context", "2048",
            "--default-max-tokens", str(MAXTOK),
            "--temperature", "0", "--spec", "mtp", "--draft-tokens", "4", "--lm-head-draft",
            "--request-log-jsonl", req_log]
    log("LAUNCH env=%s" % env_extra)
    sf = open(serve_log, "w", encoding="utf-8")
    # start_new_session detaches the child into its own process group so it is not killed
    # by terminal job-control; we manage its lifetime explicitly.
    proc = subprocess.Popen(args, stdout=sf, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    sf.close()
    # Wait for the port to bind, then for the model to actually load.
    deadline = time.time() + 420
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("serve exited early rc=%s; see %s" % (proc.returncode, serve_log))
        if port_open(PORT):
            break
        time.sleep(2)
    else:
        raise RuntimeError("serve did not bind :%d; see %s" % (PORT, serve_log))
    ready_deadline = time.time() + 360
    while time.time() < ready_deadline:
        if proc.poll() is not None:
            raise RuntimeError("serve died during model load rc=%s; see %s" % (proc.returncode, serve_log))
        if model_ready():
            log("serve ready :%d pid=%s" % (PORT, proc.pid))
            return proc
        time.sleep(3)
    raise RuntimeError("model not ready within 360s; see %s" % serve_log)


def stop_serve(proc):
    if proc.poll() is not None:
        return
    log("stopping serve pid=%s" % proc.pid)
    try:
        # docs: first SIGINT shows a confirm prompt; a second within 5s stops the server.
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        time.sleep(1)
        os.killpg(os.getpgid(proc.pid), signal.SIGINT)
    except ProcessLookupError:
        return
    for _ in range(40):
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except Exception:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except Exception:
            pass


def get_model_id():
    try:
        with urllib.request.urlopen("http://%s:%d/v1/models" % (HOST, PORT), timeout=10) as r:
            return json.loads(r.read().decode())["data"][0]["id"]
    except Exception:
        return "qwen3.8-flash"


def http_chat(model_id, content, max_tokens):
    body = {"model": model_id, "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.0, "stream": False}
    req = urllib.request.Request("http://%s:%d/v1/chat/completions" % (HOST, PORT),
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=1200) as resp:
        return json.loads(resp.read().decode())


def burst(model_id):
    """Fire all prompts concurrently; return (wall_s, [{prompt_idx, completion_tokens, text}])."""
    results = [None] * len(PROMPTS)

    def one(i):
        t0 = time.time()
        data = http_chat(model_id, PROMPTS[i], MAXTOK)
        ch = data.get("choices", [{}])[0]
        msg = ch.get("message", {}) or {}
        return {"prompt_idx": i, "completion_tokens": (data.get("usage") or {}).get("completion_tokens"),
                "text": msg.get("content", ""), "finish": ch.get("finish_reason"), "client_s": time.time() - t0}

    wall0 = time.time()
    with ThreadPoolExecutor(max_workers=len(PROMPTS)) as ex:
        futs = {i: ex.submit(one, i) for i in range(len(PROMPTS))}
        for i, f in futs.items():
            results[i] = f.result()
    wall = time.time() - wall0
    return wall, results


def smoke(model_id):
    try:
        d = http_chat(model_id, "Reply with the single word: warm", 8)
        log("smoke finish=%s" % d.get("choices", [{}])[0].get("finish_reason"))
    except Exception as e:
        log("smoke error: %s" % e)


def parse_log(path):
    """Return request_done rows and the peak/last decode_batch co-scheduling stats."""
    rows, throughput = [], []
    if not os.path.exists(path):
        return rows, throughput
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except Exception:
                continue
            ev = o.get("event")
            if ev == "request_done":
                res = o.get("result") or {}
                t = o.get("timings_seconds") or {}
                req = o.get("request") or {}
                if (req.get("message_count") or 0) <= 1:
                    continue
                rows.append({"completion_tokens": res.get("completion_tokens"),
                             "decode_s": t.get("decode"), "total_s": t.get("total"),
                             "finish": res.get("finish_reason")})
            elif ev == "throughput":
                db = o.get("decode_batch") or {}
                throughput.append({"row_rounds": db.get("row_rounds"), "rounds": db.get("rounds"),
                                   "avg_size": db.get("average_size")})
    return rows, throughput


def run_arm(arm_name, env_extra):
    serve_log = os.path.join(AB_DIR, "serve_%s.log" % arm_name)
    req_log = os.path.join(AB_DIR, "log_%s.jsonl" % arm_name)
    proc = start_serve(env_extra, serve_log, req_log)
    try:
        mid = get_model_id()
        smoke(mid)
        agg_tok_s_list = []; row_rounds_max = 0; avg_size_max = 0.0
        transcripts_all = []
        for r in range(REPEATS):
            wall, res = burst(mid)
            total = sum(x["completion_tokens"] or 0 for x in res)
            agg = total / wall if wall > 0 else 0.0
            agg_tok_s_list.append(agg)
            transcripts_all.append([x["text"] for x in res])
            log("  %s leg%d: wall=%.2fs total_tok=%d aggregate=%.2f tok/s" % (arm_name, r, wall, total, agg))
        rows, throughput = parse_log(req_log)
        for th in throughput:
            if th["row_rounds"]:
                row_rounds_max = max(row_rounds_max, th["row_rounds"])
            if th["avg_size"]:
                avg_size_max = max(avg_size_max, th["avg_size"])
        return {"arm": arm_name, "agg_tok_s_list": agg_tok_s_list,
                "agg_tok_s_best": max(agg_tok_s_list) if agg_tok_s_list else 0.0,
                "agg_tok_s_mean": (sum(agg_tok_s_list) / len(agg_tok_s_list)) if agg_tok_s_list else 0.0,
                "row_rounds_max": row_rounds_max, "avg_size_max": avg_size_max,
                "n_done": len(rows), "transcripts": transcripts_all}
    finally:
        stop_serve(proc)
        settle_gpu()


def main():
    # Save transcripts per arm for the bit-exactness diff.
    arms = {}
    arms["OFF"] = run_arm("OFF", {"INFERNIX_MULTIROW_DRAFTS": "0.0"})
    arms["ON"] = run_arm("ON", {"INFERNIX_MULTIROW_DRAFTS": "1.0"})

    with open(RESULTS, "w", encoding="utf-8") as f:
        f.write("arm\trow_rounds_max\tavg_size_max\tn_done\tagg_mean\tagg_best\tlegs\n")
        for name in ("OFF", "ON"):
            a = arms[name]
            f.write("%s\t%s\t%s\t%s\t%.2f\t%.2f\t%s\n" % (
                name, a["row_rounds_max"], a["avg_size_max"], a["n_done"],
                a["agg_tok_s_mean"], a["agg_tok_s_best"],
                ",".join("%.2f" % v for v in a["agg_tok_s_list"])))

    # Bit-exactness gate: identical prompt text across arms must yield identical greedy output.
    exact = True
    for i in range(len(PROMPTS)):
        off = [arms["OFF"]["transcripts"][leg][i] for leg in range(REPEATS)]
        on = [arms["ON"]["transcripts"][leg][i] for leg in range(REPEATS)]
        # every leg within an arm should also match (greedy determinism check)
        if len(set(off)) != 1 or len(set(on)) != 1:
            log("BIT-EXACT: prompt %d not deterministic within an arm" % i)
            exact = False
        if off[0] != on[0]:
            log("BIT-EXACT: prompt %d OFF != ON (len %d vs %d)" % (i, len(off[0]), len(on[0])))
            exact = False
    log("BIT-EXACT GATE: %s" % ("PASS (speculative verify preserved greedy output)" if exact else "FAIL"))

    off, on = arms["OFF"], arms["ON"]
    delta = ((on["agg_tok_s_mean"] - off["agg_tok_s_mean"]) / off["agg_tok_s_mean"] * 100.0) if off["agg_tok_s_mean"] else 0.0
    log("AGGREGATE tok/s  OFF=%.2f  ON=%.2f  delta=%+.1f%%" % (off["agg_tok_s_mean"], on["agg_tok_s_mean"], delta))
    log("co-scheduling row_rounds  OFF=%s  ON=%s" % (off["row_rounds_max"], on["row_rounds_max"]))
    print("\nRESULTS written to %s" % RESULTS)
    return 0


if __name__ == "__main__":
    sys.exit(main())
