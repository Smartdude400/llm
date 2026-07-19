/*  gpu_ref.c - CPU stand-in for gpu_mac.m.
 *
 *  Same interface, plain math. Its real job is testing: it enforces exactly
 *  the discipline the Metal backend requires, namely that every matrix handed
 *  to sgemm lives inside a buffer that was registered first. On Linux nothing
 *  technically needs registering, but if tinyllm.c forgets to register an
 *  allocation this aborts here instead of misbehaving on a Mac we cannot test.
 *
 *      cc -O3 -DUSE_MPS -o tinyllm tinyllm.c gpu_ref.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "tl_gpu.h"

#define MAX_BUFS 256

typedef struct { const char *lo, *hi; } Range;
static Range g_r[MAX_BUFS];
static int g_n = 0;
static long g_calls = 0;
static int g_ready = 0;

int tl_gpu_init(void) { g_ready = 1; return 0; }
const char *tl_gpu_name(void) { return "cpu-reference"; }

void tl_gpu_register(void *ptr, long bytes) {
    if (!g_ready) {
        fprintf(stderr, "gpu_ref: tl_gpu_register called before tl_gpu_init - "
                        "Metal would have dropped this buffer silently\n");
        exit(1);
    }
    if (g_n >= MAX_BUFS) {
        fprintf(stderr, "gpu_ref: too many registered buffers\n");
        exit(1);
    }
    g_r[g_n].lo = (const char *)ptr;
    g_r[g_n].hi = (const char *)ptr + bytes;
    g_n++;
}

static void require_registered(const void *p, const char *what) {
    for (int i = 0; i < g_n; i++)
        if ((const char *)p >= g_r[i].lo && (const char *)p < g_r[i].hi) return;
    fprintf(stderr, "gpu_ref: matrix %s at %p was never registered - "
                    "the Metal backend could not use it\n", what, p);
    exit(1);
}

void tl_gpu_sgemm(int transA, int transB,
                  int M, int N, int K,
                  float alpha, const float *A, int lda,
                              const float *B, int ldb,
                  float beta,        float *C, int ldc) {
    require_registered(A, "A");
    require_registered(B, "B");
    require_registered(C, "C");
    g_calls++;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                float a = transA ? A[(long)k * lda + i] : A[(long)i * lda + k];
                float b = transB ? B[(long)j * ldb + k] : B[(long)k * ldb + j];
                acc += a * b;
            }
            float *c = &C[(long)i * ldc + j];
            *c = (beta == 0.0f) ? alpha * acc : alpha * acc + beta * (*c);
        }
    }
}

void tl_gpu_sync(void) { }

void tl_gpu_shutdown(void) {
    if (getenv("TINYLLM_GPU_STATS"))
        fprintf(stderr, "gpu_ref: %ld sgemm calls, %d registered buffers\n",
                g_calls, g_n);
    g_n = 0;
}
