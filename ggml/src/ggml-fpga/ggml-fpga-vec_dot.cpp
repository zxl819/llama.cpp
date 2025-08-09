#include "ggml-fpga.h"
#include "ggml-cpu/vec.h"
#include "ggml-cpu/quants.h"

#ifdef GGML_USE_FPGA
#undef ggml_vec_dot
#endif

static ggml_fpga_vec_dot_t fpga_vec_dot_impl = nullptr;

extern "C" {

void ggml_fpga_set_vec_dot(ggml_fpga_vec_dot_t fn) {
    fpga_vec_dot_impl = fn;
}

static inline void ggml_vec_dot_cpu_fallback(int n, float * s, size_t bs,
                                             const void * x, size_t bx,
                                             const void * y, size_t by, int nrc) {
    ggml_vec_dot_q8_0_q8_0(n, s, bs, x, bx, y, by, nrc);
}

void ggml_vec_dot_fpga(int n, float * s, size_t bs,
                       const void * x, size_t bx,
                       const void * y, size_t by, int nrc) {
    if (fpga_vec_dot_impl && fpga_vec_dot_impl(n, s, bs, x, bx, y, by, nrc)) {
        return;
    }
    ggml_vec_dot_cpu_fallback(n, s, bs, x, bx, y, by, nrc);
}

} // extern "C"
