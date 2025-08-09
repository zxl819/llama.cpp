#include "ggml-fpga.h"

#ifdef GGML_USE_FPGA
#undef ggml_mul_mat
#endif

static ggml_fpga_mul_mat_t fpga_mul_mat_impl = nullptr;

extern "C" {

void ggml_fpga_set_mul_mat(ggml_fpga_mul_mat_t fn) {
    fpga_mul_mat_impl = fn;
}

static inline struct ggml_tensor * ggml_mul_mat_cpu_fallback(struct ggml_context * ctx,
                                                            struct ggml_tensor * a,
                                                            struct ggml_tensor * b) {
    return ggml_mul_mat(ctx, a, b);
}

struct ggml_tensor * ggml_mul_mat_fpga(struct ggml_context * ctx,
                                       struct ggml_tensor * a,
                                       struct ggml_tensor * b) {
    if (fpga_mul_mat_impl &&
        a->type == GGML_TYPE_Q8_0 && b->type == GGML_TYPE_Q8_0) {
        if (struct ggml_tensor * out = fpga_mul_mat_impl(ctx, a, b)) {
            return out;
        }
    }
    return ggml_mul_mat_cpu_fallback(ctx, a, b);
}

} // extern "C"
