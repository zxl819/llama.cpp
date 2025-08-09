#include "ggml-fpga.h"

#ifdef GGML_USE_FPGA
#undef ggml_rms_norm
#endif

static ggml_fpga_rms_norm_t fpga_rms_norm_impl = nullptr;

extern "C" {

void ggml_fpga_set_rms_norm(ggml_fpga_rms_norm_t fn) {
    fpga_rms_norm_impl = fn;
}

static inline struct ggml_tensor * ggml_rms_norm_cpu_fallback(struct ggml_context * ctx,
                                                             struct ggml_tensor * a,
                                                             float eps) {
    return ggml_rms_norm(ctx, a, eps);
}

struct ggml_tensor * ggml_rms_norm_fpga(struct ggml_context * ctx,
                                       struct ggml_tensor * a,
                                       float eps) {
    if (fpga_rms_norm_impl) {
        if (struct ggml_tensor * out = fpga_rms_norm_impl(ctx, a, eps)) {
            return out;
        }
    }
    return ggml_rms_norm_cpu_fallback(ctx, a, eps);
}

} // extern "C"
