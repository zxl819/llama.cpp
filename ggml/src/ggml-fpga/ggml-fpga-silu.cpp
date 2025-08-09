#include "ggml-fpga.h"

#ifdef GGML_USE_FPGA
#undef ggml_silu
#endif

static ggml_fpga_silu_t fpga_silu_impl = nullptr;

extern "C" {

void ggml_fpga_set_silu(ggml_fpga_silu_t fn) {
    fpga_silu_impl = fn;
}

static inline struct ggml_tensor * ggml_silu_cpu_fallback(struct ggml_context * ctx,
                                                          struct ggml_tensor * a) {
    return ggml_silu(ctx, a);
}

struct ggml_tensor * ggml_silu_fpga(struct ggml_context * ctx,
                                    struct ggml_tensor * a) {
    if (fpga_silu_impl) {
        if (struct ggml_tensor * out = fpga_silu_impl(ctx, a)) {
            return out;
        }
    }
    return ggml_silu_cpu_fallback(ctx, a);
}

} // extern "C"
