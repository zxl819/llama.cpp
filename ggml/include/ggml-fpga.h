#pragma once

#include "ggml.h"

#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Types for FPGA-accelerated operations. Implementations should return true on
// success.  Both matrix operands and the vectors are expected to contain
// GGML_TYPE_Q8_0 data (int8 with per-block scale).
typedef struct ggml_tensor * (*ggml_fpga_mul_mat_t)(struct ggml_context * ctx,
                                                   struct ggml_tensor * a,
                                                   struct ggml_tensor * b);

typedef bool (*ggml_fpga_vec_dot_t)(int n, float * s, size_t bs,
                                    const void * x, size_t bx,
                                    const void * y, size_t by, int nrc);

typedef struct ggml_tensor * (*ggml_fpga_rms_norm_t)(struct ggml_context * ctx,
                                                     struct ggml_tensor * a,
                                                     float eps);

typedef struct ggml_tensor * (*ggml_fpga_silu_t)(struct ggml_context * ctx,
                                                 struct ggml_tensor * a);

// Registration helpers allowing applications to provide FPGA implementations
void ggml_fpga_set_mul_mat(ggml_fpga_mul_mat_t fn);
void ggml_fpga_set_vec_dot(ggml_fpga_vec_dot_t fn);
void ggml_fpga_set_rms_norm(ggml_fpga_rms_norm_t fn);
void ggml_fpga_set_silu(ggml_fpga_silu_t fn);

// Wrappers that call the registered FPGA implementations and fall back to CPU
struct ggml_tensor * ggml_mul_mat_fpga(struct ggml_context * ctx,
                                       struct ggml_tensor * a,
                                       struct ggml_tensor * b);
void ggml_vec_dot_fpga(int n, float * s, size_t bs,
                       const void * x, size_t bx,
                       const void * y, size_t by, int nrc);
struct ggml_tensor * ggml_rms_norm_fpga(struct ggml_context * ctx,
                                       struct ggml_tensor * a,
                                       float eps);
struct ggml_tensor * ggml_silu_fpga(struct ggml_context * ctx,
                                    struct ggml_tensor * a);

// Optional macro remapping so existing call sites can transparently use the
// FPGA implementations when GGML_USE_FPGA is defined.
#ifdef GGML_USE_FPGA
#define ggml_mul_mat ggml_mul_mat_fpga
#define ggml_vec_dot ggml_vec_dot_fpga
#define ggml_rms_norm ggml_rms_norm_fpga
#define ggml_silu ggml_silu_fpga
#endif

#ifdef __cplusplus
}
#endif
