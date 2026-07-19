/*  gpu_mac.m - Metal Performance Shaders backend for tinyllm.
 *
 *  Build (Apple Silicon):
 *      clang -O3 -ffast-math -DUSE_MPS -DCTX=128 -DBATCH=16 \
 *            -pthread -DNTHREADS=4 -fobjc-arc \
 *            -framework Metal -framework MetalPerformanceShaders \
 *            -framework Foundation \
 *            -o tinyllm-gpu tinyllm.c gpu_mac.m
 *
 *  Why this can be fast: Apple Silicon has unified memory, so a page-aligned
 *  host allocation can be wrapped as an MTLBuffer with newBufferWithBytesNoCopy
 *  and the GPU reads the very same bytes. No upload, no download. Without that
 *  trick a model this small would spend all its time on PCIe-style copies and
 *  lose badly to the CPU.
 *
 *  Command buffers are batched: work is encoded as it arrives and only
 *  committed when tl_gpu_sync() is called, so a whole forward or backward
 *  pass becomes a few submissions rather than one per matmul.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include "tl_gpu.h"

#define MAX_BUFS 256
#define MAX_PENDING 64          /* encode this many GEMMs before committing */

static id<MTLDevice> g_dev = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLCommandBuffer> g_cb = nil;
static int g_pending = 0;
static char g_name[256] = "none";

/* registry of host pointers we have wrapped as device buffers */
typedef struct {
    const void *base;
    long bytes;
    id<MTLBuffer> buf;
} BufEntry;

static BufEntry g_bufs[MAX_BUFS];
static int g_nbufs = 0;

int tl_gpu_init(void) {
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) {
            fprintf(stderr, "Metal: no default device\n");
            return 1;
        }
        g_queue = [g_dev newCommandQueue];
        if (!g_queue) {
            fprintf(stderr, "Metal: could not create a command queue\n");
            g_dev = nil;
            return 2;
        }
        snprintf(g_name, sizeof g_name, "%s", [[g_dev name] UTF8String]);
        return 0;
    }
}

const char *tl_gpu_name(void) { return g_name; }

void tl_gpu_register(void *ptr, long bytes) {
    if (!g_dev || g_nbufs >= MAX_BUFS || !ptr || bytes <= 0) return;
    /* newBufferWithBytesNoCopy needs page alignment and a page-multiple length */
    long pagesize = (long)getpagesize();
    if (((uintptr_t)ptr % (uintptr_t)pagesize) != 0) {
        fprintf(stderr, "Metal: pointer %p not page-aligned; "
                        "it will fall back to a copy\n", ptr);
        return;
    }
    long rounded = ((bytes + pagesize - 1) / pagesize) * pagesize;
    id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:ptr
                                               length:(NSUInteger)rounded
                                              options:MTLResourceStorageModeShared
                                          deallocator:nil];
    if (!b) {
        fprintf(stderr, "Metal: could not wrap %ld bytes at %p\n", bytes, ptr);
        return;
    }
    g_bufs[g_nbufs].base = ptr;
    g_bufs[g_nbufs].bytes = rounded;
    g_bufs[g_nbufs].buf = b;
    g_nbufs++;
}

/* find the registered buffer containing p, and p's offset within it */
static id<MTLBuffer> find_buf(const void *p, NSUInteger *offset) {
    for (int i = 0; i < g_nbufs; i++) {
        const char *lo = (const char *)g_bufs[i].base;
        const char *hi = lo + g_bufs[i].bytes;
        if ((const char *)p >= lo && (const char *)p < hi) {
            *offset = (NSUInteger)((const char *)p - lo);
            return g_bufs[i].buf;
        }
    }
    return nil;
}

static void flush_commands(int wait) {
    if (!g_cb) return;
    [g_cb commit];
    if (wait) [g_cb waitUntilCompleted];
    g_cb = nil;
    g_pending = 0;
}

void tl_gpu_sgemm(int transA, int transB,
                  int M, int N, int K,
                  float alpha, const float *A, int lda,
                              const float *B, int ldb,
                  float beta,        float *C, int ldc) {
    @autoreleasepool {
        NSUInteger offA = 0, offB = 0, offC = 0;
        id<MTLBuffer> bufA = find_buf(A, &offA);
        id<MTLBuffer> bufB = find_buf(B, &offB);
        id<MTLBuffer> bufC = find_buf(C, &offC);
        if (!bufA || !bufB || !bufC) {
            /* Not registered: we cannot run this on the GPU. Refuse loudly
             * rather than silently producing wrong numbers. */
            fprintf(stderr, "Metal: unregistered matrix (A=%p B=%p C=%p)\n",
                    (void *)A, (void *)B, (void *)C);
            return;
        }

        MPSMatrixDescriptor *dA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)(transA ? K : M)
                             columns:(NSUInteger)(transA ? M : K)
                            rowBytes:(NSUInteger)lda * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)(transB ? N : K)
                             columns:(NSUInteger)(transB ? K : N)
                            rowBytes:(NSUInteger)ldb * sizeof(float)
                            dataType:MPSDataTypeFloat32];
        MPSMatrixDescriptor *dC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(NSUInteger)M
                             columns:(NSUInteger)N
                            rowBytes:(NSUInteger)ldc * sizeof(float)
                            dataType:MPSDataTypeFloat32];

        MPSMatrix *mA = [[MPSMatrix alloc] initWithBuffer:bufA offset:offA descriptor:dA];
        MPSMatrix *mB = [[MPSMatrix alloc] initWithBuffer:bufB offset:offB descriptor:dB];
        MPSMatrix *mC = [[MPSMatrix alloc] initWithBuffer:bufC offset:offC descriptor:dC];

        MPSMatrixMultiplication *op = [[MPSMatrixMultiplication alloc]
            initWithDevice:g_dev
             transposeLeft:(transA ? YES : NO)
            transposeRight:(transB ? YES : NO)
                resultRows:(NSUInteger)M
             resultColumns:(NSUInteger)N
           interiorColumns:(NSUInteger)K
                     alpha:(double)alpha
                      beta:(double)beta];

        if (!g_cb) g_cb = [g_queue commandBuffer];
        [op encodeToCommandBuffer:g_cb leftMatrix:mA rightMatrix:mB resultMatrix:mC];
        g_pending++;

        /* MPS reads C when beta != 0, so consecutive accumulations into the
         * same C must not be reordered; the command buffer preserves order.
         * We only flush to bound latency and memory. */
        if (g_pending >= MAX_PENDING) flush_commands(0);
    }
}

void tl_gpu_sync(void) {
    @autoreleasepool { flush_commands(1); }
}

void tl_gpu_shutdown(void) {
    @autoreleasepool {
        flush_commands(1);
        for (int i = 0; i < g_nbufs; i++) g_bufs[i].buf = nil;
        g_nbufs = 0;
        g_queue = nil;
        g_dev = nil;
    }
}
