/* tl_gpu.h - one small seam between tinyllm.c and whatever does the matmuls.
 *
 * Two implementations satisfy this interface:
 *   gpu_mac.m   Metal Performance Shaders on the Apple GPU  (needs a Mac)
 *   gpu_ref.c   a plain CPU reference used to test the plumbing anywhere
 *
 * Everything is row-major, matching CBLAS conventions:
 *     C = alpha * op(A) * op(B) + beta * C
 * with op(A) = A if !transA else A-transposed.
 *   A is MxK (or KxM stored, if transA), lda = stride in floats
 *   B is KxN (or NxK stored, if transB), ldb = stride in floats
 *   C is MxN,                            ldc = stride in floats
 */
#ifndef TL_GPU_H
#define TL_GPU_H

/* Returns 0 on success, non-zero if no usable device was found.
 * On failure the caller should fall back to the CPU path. */
int  tl_gpu_init(void);

/* Human-readable device name, or "none". Never NULL. */
const char *tl_gpu_name(void);

/* Register a host allocation so it can be shared with the device without a
 * copy. On Apple Silicon memory is unified, so a page-aligned pointer can be
 * wrapped directly. Call once per buffer, right after allocating it. */
void tl_gpu_register(void *ptr, long bytes);

void tl_gpu_sgemm(int transA, int transB,
                  int M, int N, int K,
                  float alpha, const float *A, int lda,
                              const float *B, int ldb,
                  float beta,        float *C, int ldc);

/* Block until queued device work is finished and results are visible to the
 * CPU. Called before any host code reads a matrix the device just wrote. */
void tl_gpu_sync(void);

void tl_gpu_shutdown(void);

#endif /* TL_GPU_H */
