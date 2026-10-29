// main.cu — gb10_lab entry point.
//
// One binary, every op selectable by name or --op all. Arguments:
//   --op NAME|all     which measurement(s) to run (default: all)
//   --preset NAME     model shape preset (default: qwen3_dense_h5120)
//   --warmup N        warm-up iterations before timing (default: 3)
//   --iters N         timed repetitions, median/p10/p90 reported (default: 20)
//   --quick           shrink sizes/iterations for smoke runs
//   --output PATH     append JSONL records here (empty = stderr)
#include "gb10_lab/common.h"
#include <cstring>
#include <cstdio>
#include <cstdlib>

// Op entry points defined across ops/*.cu.
void run_fingerprint(const LabArgs&);
void run_memory(const LabArgs&);
void run_compute(const LabArgs&);
void run_rmsnorm(const LabArgs&);
void run_rope(const LabArgs&);
void run_softmax(const LabArgs&);
void run_gemv(const LabArgs&);
void run_dense_width(const LabArgs&);
void run_tensor_fp4_gemv(const LabArgs&);
void run_int_dot(const LabArgs&);
void run_attention_decode(const LabArgs&);
void run_layer_pipeline(const LabArgs&);

namespace {
struct OpDef { const char* name; void (*fn)(const LabArgs&); };
constexpr OpDef kOps[] = {
    {"fingerprint",      run_fingerprint},
    {"memory",           run_memory},
    {"compute",          run_compute},
    {"rmsnorm",          run_rmsnorm},
    {"rope",             run_rope},
    {"softmax",          run_softmax},
    {"gemv",             run_gemv},
    {"dense_width",      run_dense_width},
    {"tensor_fp4_gemv",  run_tensor_fp4_gemv},
    {"int_dot",          run_int_dot},
    {"attention_decode", run_attention_decode},
    {"layer_pipeline",   run_layer_pipeline},
};

bool parse_int(const char* v, int& out) {
    char* end = nullptr; long x = std::strtol(v, &end, 10);
    if (!end || *end != '\0') return false; out = (int)x; return true;
}
void usage() {
    std::fprintf(stderr,
        "usage: gb10_bench [--op NAME|all] [--preset NAME] [--warmup N]\n"
        "                  [--iters N] [--quick] [--sustained N] [--output PATH]\n"
        "ops: fingerprint memory compute rmsnorm rope softmax gemv dense_width\n"
        "     int_dot attention_decode layer_pipeline\n"
        "--sustained N: keep GPU continuously fed (N launches/iter) to raise\n"
        "               power/util toward a real workload steady state.\n");
}
} // namespace

int main(int argc, char** argv) {
    const char* op_sel = "all";
    const char* preset_name = "qwen3_dense_h5120";
    int warmup = 3, iters = 20;
    bool quick = false;
    const char* output = "";
    int sustained = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](const char*& dst){ if (i+1 >= argc){ std::fprintf(stderr,"missing value for %s\n",a.c_str()); std::exit(2);} dst = argv[++i]; };
        if      (a == "--op")     { const char* v; next(v); op_sel = v; }
        else if (a == "--preset") { const char* v; next(v); preset_name = v; }
        else if (a == "--warmup") { const char* v; next(v); if(!parse_int(v,warmup)){std::fprintf(stderr,"bad --warmup\n");return 2;} }
        else if (a == "--iters")  { const char* v; next(v); if(!parse_int(v,iters)){std::fprintf(stderr,"bad --iters\n");return 2;} }
        else if (a == "--quick")  { quick = true; }
        else if (a == "--sustained") { const char* v; next(v); if(!parse_int(v,sustained)){std::fprintf(stderr,"bad --sustained\n");return 2;} }
        else if (a == "--output") { const char* v; next(v); output = v; }
        else if (a == "--help" || a == "-h") { usage(); return 0; }
        else { std::fprintf(stderr,"unknown argument '%s'\n", a.c_str()); usage(); return 2; }
    }
    g_sustained = sustained;
    json_out_path().assign(output ? output : "");

    const ModelPreset& preset = preset_by_name(preset_name);
    LabArgs args{preset, warmup, iters, quick, output ? output : ""};

    DeviceInfo dev = DeviceInfo::get();
    std::printf("gb10_lab | device=%s cc=%d.%d sm=%d l2=%.1fMiB | preset=%s warmup=%d iters=%d quick=%d\n",
                dev.name.c_str(), dev.cc_major, dev.cc_minor, dev.sm_count,
                dev.l2_bytes / (1024.0*1024.0), args.preset.name, warmup, iters, (int)quick);

    bool any = false, failed = false;
    for (const auto& op : kOps) {
        if (std::strcmp(op_sel, "all") != 0 && std::strcmp(op_sel, op.name) != 0) continue;
        any = true;
        std::printf("\n== %s ==\n", op.name);
        try {
            op.fn(args);
        } catch (const std::exception& e) {
            std::fprintf(stderr, "op '%s' threw: %s\n", op.name, e.what());
            failed = true;
        }
    }
    if (!any) { std::fprintf(stderr, "no matching op '%s'\n", op_sel); usage(); return 2; }
    return failed ? 1 : 0;
}
