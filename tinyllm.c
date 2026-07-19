/*  tinyllm.c — a tiny byte-level GPT: train it on a Mac, a Linux box, or a
 *  phone. Same file, same checkpoint format everywhere.
 *
 *  2 transformer layers, 4 heads, width 64, context 64 by default. The vocab
 *  is built from the bytes that actually occur in your training file (~65 for
 *  English prose, ~88 for Python) instead of a fixed 256. Hand-written
 *  forward + backward + AdamW, checkpointing with auto-resume, sampling, and
 *  a finite-difference gradient check.
 *
 *  ---- macOS (fastest: Accelerate drives the AMX matrix unit) -------------
 *      ./build.sh                       # detects cores, picks flags
 *  or by hand:
 *      cc -O3 -ffast-math -DUSE_BLAS -DCTX=128 -DBATCH=16 \
 *         -pthread -DNTHREADS=4 -framework Accelerate -o tinyllm tinyllm.c
 *
 *  ---- Linux -------------------------------------------------------------
 *      cc -O3 -ffast-math -DUSE_BLAS -DCTX=128 -DBATCH=16 \
 *         -pthread -DNTHREADS=4 -o tinyllm tinyllm.c -lopenblas -lm
 *      (drop -DUSE_BLAS and -lopenblas if you don't have OpenBLAS)
 *
 *  ---- iOS ---------------------------------------------------------------
 *      iSH:      gcc -O3 -o tinyllm tinyllm.c -lm
 *      a-Shell:  clang -O3 -o tinyllm.wasm tinyllm.c   (no threads/BLAS there)
 *
 *  ---- use ---------------------------------------------------------------
 *      ./tinyllm train data.txt model.bin 4000   # resumes if model.bin exists
 *      ./tinyllm gen   model.bin "def " 400 0.5
 *      ./tinyllm check data.txt                  # verify the backward pass
 *
 *  Every knob is -D overridable: DIM, NLAYER, NHEAD, CTX, BATCH, LEARN_RATE,
 *  NTHREADS. BATCH must be divisible by NTHREADS. Use CTX=128 for source
 *  code — 64 characters is about one line, too short to track indentation.
 *
 *  The printed tok/s is your real speed: one step = BATCH*CTX tokens, so
 *  steps_needed * BATCH * CTX / (tok/s) = seconds. The checkpoint stores the
 *  config + charset and refuses to load into a mismatched build, so changing
 *  CTX or DIM means starting a fresh model.bin.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

/* ---------------------------------------------------------------------
 * ALL system headers must be included BEFORE this file's #defines.
 * Apple's Accelerate pulls in CarbonCore, whose MachineExceptions.h has a
 * struct field literally named LR -- so a `#define LR` above this point
 * rewrites it to `UnsignedWide 1e-3f;` and the SDK fails to parse. Short
 * macro names and system headers do not mix; keep the includes on top.
 * ------------------------------------------------------------------ */
#ifndef NTHREADS
#define NTHREADS 1          /* compile with -pthread -DNTHREADS=2 (or 4) */
#endif
#if NTHREADS > 1
#include <pthread.h>
#endif

/* Optional BLAS for the matmuls. On macOS this is Apple's Accelerate, which
 * uses the AMX matrix coprocessor on Apple Silicon:
 *     cc -O3 -DUSE_BLAS -framework Accelerate -o tinyllm tinyllm.c
 * On Linux the identical CBLAS API comes from OpenBLAS:
 *     cc -O3 -DUSE_BLAS -o tinyllm tinyllm.c -lopenblas -lm            */
#ifdef USE_BLAS
#ifdef __APPLE__
/* Apple deprecated the classic CBLAS prototypes in macOS 13.3; this selects
 * the supported headers. Drop it if you are on an older macOS. */
#ifndef ACCELERATE_NEW_LAPACK
#define ACCELERATE_NEW_LAPACK 1
#endif
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif
#endif

/* ------------------------- model size knobs ------------------------- */
/* All knobs are -D overridable: cc -O3 -DCTX=128 -DBATCH=16 ... */
#ifndef DIM
#define DIM     64          /* model width (embedding size)             */
#endif
#ifndef NLAYER
#define NLAYER  2           /* transformer blocks                       */
#endif
#ifndef NHEAD
#define NHEAD   4           /* attention heads, DIM % NHEAD must be 0   */
#endif
#ifndef CTX
#define CTX     64          /* context length in tokens (bytes)         */
#endif
#define FF      (4*DIM)     /* MLP hidden size                          */
#define HD      (DIM/NHEAD) /* per-head size                            */
#define MAXV    256         /* upper bound on vocab (bytes)             */

/* -------------------------- training knobs ------------------------- */
#ifndef BATCH
#define BATCH        4
#endif
/* Named in full on purpose: a command-line -DLR would be live while the
 * system headers are parsed, and Apple's MachineExceptions.h has a struct
 * field called LR. Short knob names are a landmine; long ones are free. */
#ifndef LEARN_RATE
#define LEARN_RATE   1e-3f
#endif
#ifndef WEIGHT_DECAY
#define WEIGHT_DECAY 1e-4f  /* AdamW weight decay                       */
#endif
#define CKPT_EVERY   100    /* save checkpoint every N steps            */
#define SAMPLE_EVERY 250    /* print a text sample every N steps        */
#define RMS_EPS      1e-5f

_Static_assert(BATCH % NTHREADS == 0, "BATCH must be divisible by NTHREADS");

/* Monotonic wall clock: correct with threads, unlike clock().
 * clock_gettime needs macOS 10.12+ (and is hidden under strict -std=c11),
 * so fall back to gettimeofday, which exists on every Mac ever shipped. */
static double now_s(void) {
#if defined(CLOCK_MONOTONIC)
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * (double)tv.tv_usec;
#endif
}

/* ------------------------- charset (vocab) -------------------------- */
static int VS = 0;                    /* vocab size = distinct bytes    */
static unsigned char enc[MAXV];       /* byte -> token id (absent -> 0) */
static unsigned char dec_[MAXV];      /* token id -> byte               */

static void build_charset(const unsigned char *d, long n) {
    int seen[MAXV] = {0};
    for (long i = 0; i < n; i++) seen[d[i]] = 1;
    VS = 0;
    memset(enc, 0, MAXV);
    for (int b = 0; b < MAXV; b++)
        if (seen[b]) { dec_[VS] = (unsigned char)b; enc[b] = (unsigned char)VS; VS++; }
    if (VS < 2) { fprintf(stderr, "data needs at least 2 distinct bytes\n"); exit(1); }
}

/* ------------------------------ rng -------------------------------- */
static unsigned long long rngs = 88172645463325252ULL;
static unsigned long long xr(void) {
    rngs ^= rngs << 13; rngs ^= rngs >> 7; rngs ^= rngs << 17;
    return rngs;
}
static float frand(void) { return (float)(xr() >> 40) / 16777216.0f; } /* [0,1) */
static float nrand(void) {                                    /* Box-Muller */
    float u1 = frand() + 1e-9f, u2 = frand();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

/* --------------------------- parameters ---------------------------- */
typedef struct {
    float *tok;             /* VS*DIM   token embeddings                */
    float *pos;             /* CTX*DIM  learned position embeddings     */
    float *rms1[NLAYER];    /* DIM      pre-attention RMSNorm weight    */
    float *wqkv[NLAYER];    /* 3*DIM x DIM  fused q,k,v projection      */
    float *wo[NLAYER];      /* DIM x DIM    attention output projection */
    float *rms2[NLAYER];    /* DIM      pre-MLP RMSNorm weight          */
    float *w1[NLAYER];      /* FF x DIM     MLP up                      */
    float *w2[NLAYER];      /* DIM x FF     MLP down                    */
    float *rmsf;            /* DIM      final RMSNorm weight            */
    float *head;            /* VS x DIM     output head                 */
} Params;

static long nparams(void) {
    long n = (long)VS*DIM + (long)CTX*DIM + DIM + (long)VS*DIM;
    n += (long)NLAYER * (DIM + 3L*DIM*DIM + (long)DIM*DIM + DIM
                         + (long)FF*DIM + (long)DIM*FF);
    return n;
}

static void map_params(Params *p, float *buf) {
    float *x = buf;
    p->tok = x; x += (long)VS*DIM;
    p->pos = x; x += (long)CTX*DIM;
    for (int l = 0; l < NLAYER; l++) {
        p->rms1[l] = x; x += DIM;
        p->wqkv[l] = x; x += 3L*DIM*DIM;
        p->wo[l]   = x; x += (long)DIM*DIM;
        p->rms2[l] = x; x += DIM;
        p->w1[l]   = x; x += (long)FF*DIM;
        p->w2[l]   = x; x += (long)DIM*FF;
    }
    p->rmsf = x; x += DIM;
    p->head = x;
}

static float *xcalloc(long n) {
    float *p = calloc((size_t)n, sizeof(float));
    if (!p) { fprintf(stderr, "out of memory (%ld floats)\n", n); exit(1); }
    return p;
}

static void init_params(float *buf) {
    Params P; map_params(&P, buf);
    long N = nparams();
    for (long i = 0; i < N; i++) buf[i] = 0.02f * nrand();
    for (long i = 0; i < (long)CTX*DIM; i++) P.pos[i] = 0.01f * nrand();
    float rs = 0.02f / sqrtf(2.0f * NLAYER);  /* scaled residual projections */
    for (int l = 0; l < NLAYER; l++) {
        for (int i = 0; i < DIM; i++) { P.rms1[l][i] = 1.0f; P.rms2[l][i] = 1.0f; }
        for (long i = 0; i < (long)DIM*DIM; i++) P.wo[l][i] = rs * nrand();
        for (long i = 0; i < (long)DIM*FF;  i++) P.w2[l][i] = rs * nrand();
    }
    for (int i = 0; i < DIM; i++) P.rmsf[i] = 1.0f;
}

/* --------------------------- activations --------------------------- */
typedef struct {
    /* forward (kept for backward) */
    float *res[NLAYER+1];   /* residual stream: res[0] = embeddings     */
    float *n1[NLAYER], *qkv[NLAYER], *att[NLAYER], *ao[NLAYER];
    float *mid[NLAYER], *n2[NLAYER], *hr[NLAYER];
    float *nf, *logits;     /* logits becomes probs in-place            */
    /* scratch */
    float *scr, *dres, *dmid, *dao, *dqkv, *dn, *dh;
} Acts;

static void alloc_acts(Acts *A, int B) {
    long BTD = (long)B * CTX * DIM;
    for (int l = 0; l <= NLAYER; l++) A->res[l] = xcalloc(BTD);
    for (int l = 0; l < NLAYER; l++) {
        A->n1[l]  = xcalloc(BTD);
        A->qkv[l] = xcalloc(3*BTD);
        A->att[l] = xcalloc((long)B * NHEAD * CTX * CTX);
        A->ao[l]  = xcalloc(BTD);
        A->mid[l] = xcalloc(BTD);
        A->n2[l]  = xcalloc(BTD);
        A->hr[l]  = xcalloc((long)B * CTX * FF);
    }
    A->nf     = xcalloc(BTD);
    A->logits = xcalloc((long)B * CTX * VS);
    A->scr    = xcalloc(BTD);
    A->dres   = xcalloc(BTD);
    A->dmid   = xcalloc(BTD);
    A->dao    = xcalloc(BTD);
    A->dqkv   = xcalloc(3*BTD);
    A->dn     = xcalloc(BTD);
    A->dh     = xcalloc((long)B * CTX * FF);
}

/* ---------------- blocked matmul primitives (the hot loops) ---------------- */
/* Y[p] = W · X[p] for p in [0,n); W is [out][in] row-major.
 * Processing 4 positions per pass loads each weight row once per 4 outputs. */
static void mm_fwd(float *restrict Y, const float *restrict X,
                   const float *restrict W, int n, int out, int in) {
#ifdef USE_BLAS
    /* Y[n x out] = X[n x in] * W[out x in]^T */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                n, out, in, 1.0f, X, in, W, in, 0.0f, Y, out);
#else
    int p = 0;
    for (; p + 4 <= n; p += 4) {
        const float *x0 = X + (long)p*in, *x1 = x0+in, *x2 = x1+in, *x3 = x2+in;
        float *y0 = Y + (long)p*out, *y1 = y0+out, *y2 = y1+out, *y3 = y2+out;
        for (int i = 0; i < out; i++) {
            const float *w = W + (long)i*in;
            float s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            for (int j = 0; j < in; j++) {
                float wj = w[j];
                s0 += wj*x0[j]; s1 += wj*x1[j]; s2 += wj*x2[j]; s3 += wj*x3[j];
            }
            y0[i] = s0; y1[i] = s1; y2[i] = s2; y3[i] = s3;
        }
    }
    for (; p < n; p++) {
        const float *x = X + (long)p*in;
        float *y = Y + (long)p*out;
        for (int i = 0; i < out; i++) {
            const float *w = W + (long)i*in;
            float s = 0;
            for (int j = 0; j < in; j++) s += w[j]*x[j];
            y[i] = s;
        }
    }
#endif
}

/* GW += dY^T · X  (weight gradient over n positions) */
static void mm_gw(float *restrict GW, const float *restrict dY,
                  const float *restrict X, int n, int out, int in) {
#ifdef USE_BLAS
    /* GW[out x in] += dY[n x out]^T * X[n x in] */
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                out, in, n, 1.0f, dY, out, X, in, 1.0f, GW, in);
#else
    int p = 0;
    for (; p + 4 <= n; p += 4) {
        const float *x0 = X + (long)p*in, *x1 = x0+in, *x2 = x1+in, *x3 = x2+in;
        const float *d0 = dY + (long)p*out, *d1 = d0+out, *d2 = d1+out, *d3 = d2+out;
        for (int i = 0; i < out; i++) {
            float a = d0[i], b = d1[i], c = d2[i], d = d3[i];
            float *g = GW + (long)i*in;
            for (int j = 0; j < in; j++)
                g[j] += a*x0[j] + b*x1[j] + c*x2[j] + d*x3[j];
        }
    }
    for (; p < n; p++) {
        const float *x = X + (long)p*in, *dy = dY + (long)p*out;
        for (int i = 0; i < out; i++) {
            float d = dy[i];
            float *g = GW + (long)i*in;
            for (int j = 0; j < in; j++) g[j] += d*x[j];
        }
    }
#endif
}

/* dX[p] = dY[p] · W  (input gradient, overwrites dX) */
static void mm_gx(float *restrict dX, const float *restrict dY,
                  const float *restrict W, int n, int out, int in) {
#ifdef USE_BLAS
    /* dX[n x in] = dY[n x out] * W[out x in] */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, in, out, 1.0f, dY, out, W, in, 0.0f, dX, in);
#else
    memset(dX, 0, (size_t)n * in * sizeof(float));
    int p = 0;
    for (; p + 4 <= n; p += 4) {
        float *x0 = dX + (long)p*in, *x1 = x0+in, *x2 = x1+in, *x3 = x2+in;
        const float *d0 = dY + (long)p*out, *d1 = d0+out, *d2 = d1+out, *d3 = d2+out;
        for (int i = 0; i < out; i++) {
            float a = d0[i], b = d1[i], c = d2[i], d = d3[i];
            const float *w = W + (long)i*in;
            for (int j = 0; j < in; j++) {
                float wj = w[j];
                x0[j] += a*wj; x1[j] += b*wj; x2[j] += c*wj; x3[j] += d*wj;
            }
        }
    }
    for (; p < n; p++) {
        float *x = dX + (long)p*in;
        const float *dy = dY + (long)p*out;
        for (int i = 0; i < out; i++) {
            float d = dy[i];
            const float *w = W + (long)i*in;
            for (int j = 0; j < in; j++) x[j] += d*w[j];
        }
    }
#endif
}

/* ------------------------------ norms ------------------------------- */
static void rmsnorm(float *y, const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + RMS_EPS);
    for (int i = 0; i < n; i++) y[i] = w[i] * x[i] * r;
}

/* dx += grad wrt x, dw += grad wrt w, given upstream dy */
static void rmsnorm_back(float *dx, float *dw, const float *dy,
                         const float *x, const float *w, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float r = 1.0f / sqrtf(ss / n + RMS_EPS);
    float c = 0.0f;
    for (int i = 0; i < n; i++) c += dy[i] * w[i] * x[i];
    c = c * r * r * r / n;
    for (int i = 0; i < n; i++) {
        dw[i] += dy[i] * x[i] * r;
        dx[i] += dy[i] * w[i] * r - x[i] * c;
    }
}

/* ----------------------------- forward ------------------------------ */
/* ix: B*T token ids. If tg != NULL, returns mean cross-entropy over B*T.
 * Leaves softmax probs in A->logits either way. */
static float forward(Params *P, Acts *A, const unsigned char *ix,
                     const unsigned char *tg, int B, int T) {
    int BT = B * T;
    float scale = 1.0f / sqrtf((float)HD);

    for (int p = 0; p < BT; p++) {                    /* embeddings */
        const float *te = P->tok + (long)ix[p] * DIM;
        const float *pe = P->pos + (long)(p % T) * DIM;
        float *o = A->res[0] + (long)p * DIM;
        for (int i = 0; i < DIM; i++) o[i] = te[i] + pe[i];
    }

    for (int l = 0; l < NLAYER; l++) {
        float *X = A->res[l];
        for (int p = 0; p < BT; p++)                  /* pre-attention norm */
            rmsnorm(A->n1[l] + (long)p*DIM, X + (long)p*DIM, P->rms1[l], DIM);
        mm_fwd(A->qkv[l], A->n1[l], P->wqkv[l], BT, 3*DIM, DIM);

        for (int b = 0; b < B; b++)                   /* causal attention */
        for (int h = 0; h < NHEAD; h++) {
            float *att = A->att[l] + (long)(b*NHEAD + h) * T * T;
            for (int t = 0; t < T; t++) {
                const float *q = A->qkv[l] + (long)(b*T + t)*3*DIM + h*HD;
                float *arow = att + (long)t * T;
                float mx = -1e30f;
                for (int u = 0; u <= t; u++) {
                    const float *k = A->qkv[l] + (long)(b*T + u)*3*DIM + DIM + h*HD;
                    float s = 0.0f;
                    for (int i = 0; i < HD; i++) s += q[i] * k[i];
                    s *= scale;
                    arow[u] = s;
                    if (s > mx) mx = s;
                }
                float sum = 0.0f;
                for (int u = 0; u <= t; u++) { arow[u] = expf(arow[u] - mx); sum += arow[u]; }
                float inv = 1.0f / sum;
                for (int u = 0; u <= t; u++) arow[u] *= inv;
                float *o = A->ao[l] + (long)(b*T + t)*DIM + h*HD;
                for (int i = 0; i < HD; i++) o[i] = 0.0f;
                for (int u = 0; u <= t; u++) {
                    const float *v = A->qkv[l] + (long)(b*T + u)*3*DIM + 2*DIM + h*HD;
                    float a = arow[u];
                    for (int i = 0; i < HD; i++) o[i] += a * v[i];
                }
            }
        }

        mm_fwd(A->scr, A->ao[l], P->wo[l], BT, DIM, DIM);   /* proj + residual */
        for (long i = 0; i < (long)BT*DIM; i++) A->mid[l][i] = X[i] + A->scr[i];

        for (int p = 0; p < BT; p++)                  /* pre-MLP norm */
            rmsnorm(A->n2[l] + (long)p*DIM, A->mid[l] + (long)p*DIM, P->rms2[l], DIM);
        mm_fwd(A->hr[l], A->n2[l], P->w1[l], BT, FF, DIM);
        for (long i = 0; i < (long)BT*FF; i++)        /* ReLU */
            if (A->hr[l][i] < 0.0f) A->hr[l][i] = 0.0f;
        mm_fwd(A->scr, A->hr[l], P->w2[l], BT, DIM, FF);
        for (long i = 0; i < (long)BT*DIM; i++) A->res[l+1][i] = A->mid[l][i] + A->scr[i];
    }

    for (int p = 0; p < BT; p++)                      /* final norm */
        rmsnorm(A->nf + (long)p*DIM, A->res[NLAYER] + (long)p*DIM, P->rmsf, DIM);
    mm_fwd(A->logits, A->nf, P->head, BT, VS, DIM);   /* head */

    double loss = 0.0;                                /* softmax (+ loss) */
    for (int p = 0; p < BT; p++) {
        float *lg = A->logits + (long)p * VS;
        float mx = -1e30f;
        for (int i = 0; i < VS; i++) if (lg[i] > mx) mx = lg[i];
        float sum = 0.0f;
        for (int i = 0; i < VS; i++) { lg[i] = expf(lg[i] - mx); sum += lg[i]; }
        float inv = 1.0f / sum;
        for (int i = 0; i < VS; i++) lg[i] *= inv;
        if (tg) loss += -log((double)lg[tg[p]] + 1e-9);
    }
    return tg ? (float)(loss / BT) : 0.0f;
}

/* ----------------------------- backward ----------------------------- */
static void backward(Params *P, Params *G, Acts *A, const unsigned char *ix,
                     const unsigned char *tg, int B, int T) {
    int BT = B * T;
    float scale = 1.0f / sqrtf((float)HD);

    /* probs -> dlogits in place */
    float inv = 1.0f / BT;
    for (int p = 0; p < BT; p++) A->logits[(long)p*VS + tg[p]] -= 1.0f;
    for (long i = 0; i < (long)BT*VS; i++) A->logits[i] *= inv;

    /* head + final norm; A->dres accumulates d res[NLAYER] */
    mm_gw(G->head, A->logits, A->nf, BT, VS, DIM);
    mm_gx(A->dn, A->logits, P->head, BT, VS, DIM);
    memset(A->dres, 0, (long)BT * DIM * sizeof(float));
    for (int p = 0; p < BT; p++)
        rmsnorm_back(A->dres + (long)p*DIM, G->rmsf, A->dn + (long)p*DIM,
                     A->res[NLAYER] + (long)p*DIM, P->rmsf, DIM);

    for (int l = NLAYER - 1; l >= 0; l--) {
        /* MLP: res[l+1] = mid + w2·relu(w1·rmsnorm(mid)) */
        memcpy(A->dmid, A->dres, (long)BT * DIM * sizeof(float));
        mm_gw(G->w2[l], A->dres, A->hr[l], BT, DIM, FF);
        mm_gx(A->dh, A->dres, P->w2[l], BT, DIM, FF);
        for (long i = 0; i < (long)BT*FF; i++)        /* ReLU mask */
            if (A->hr[l][i] <= 0.0f) A->dh[i] = 0.0f;
        mm_gw(G->w1[l], A->dh, A->n2[l], BT, FF, DIM);
        mm_gx(A->dn, A->dh, P->w1[l], BT, FF, DIM);
        for (int p = 0; p < BT; p++)
            rmsnorm_back(A->dmid + (long)p*DIM, G->rms2[l], A->dn + (long)p*DIM,
                         A->mid[l] + (long)p*DIM, P->rms2[l], DIM);

        /* attention: mid = res[l] + wo·attn(rmsnorm(res[l])) */
        memcpy(A->dres, A->dmid, (long)BT * DIM * sizeof(float));
        mm_gw(G->wo[l], A->dmid, A->ao[l], BT, DIM, DIM);
        mm_gx(A->dao, A->dmid, P->wo[l], BT, DIM, DIM);

        memset(A->dqkv, 0, (long)BT * 3 * DIM * sizeof(float));
        for (int b = 0; b < B; b++)
        for (int h = 0; h < NHEAD; h++) {
            const float *att = A->att[l] + (long)(b*NHEAD + h) * T * T;
            for (int t = 0; t < T; t++) {
                const float *dout = A->dao + (long)(b*T + t)*DIM + h*HD;
                const float *arow = att + (long)t * T;
                float dp[CTX];
                float sum = 0.0f;
                for (int u = 0; u <= t; u++) {
                    const float *v  = A->qkv[l] + (long)(b*T + u)*3*DIM + 2*DIM + h*HD;
                    float *dv = A->dqkv + (long)(b*T + u)*3*DIM + 2*DIM + h*HD;
                    float a = arow[u], d = 0.0f;
                    for (int i = 0; i < HD; i++) { dv[i] += a * dout[i]; d += dout[i] * v[i]; }
                    dp[u] = d;
                    sum += a * d;
                }
                const float *q  = A->qkv[l] + (long)(b*T + t)*3*DIM + h*HD;
                float *dq = A->dqkv + (long)(b*T + t)*3*DIM + h*HD;
                for (int u = 0; u <= t; u++) {
                    float ds = arow[u] * (dp[u] - sum) * scale;
                    const float *k  = A->qkv[l] + (long)(b*T + u)*3*DIM + DIM + h*HD;
                    float *dk = A->dqkv + (long)(b*T + u)*3*DIM + DIM + h*HD;
                    for (int i = 0; i < HD; i++) { dq[i] += ds * k[i]; dk[i] += ds * q[i]; }
                }
            }
        }

        mm_gw(G->wqkv[l], A->dqkv, A->n1[l], BT, 3*DIM, DIM);
        mm_gx(A->dn, A->dqkv, P->wqkv[l], BT, 3*DIM, DIM);
        for (int p = 0; p < BT; p++)
            rmsnorm_back(A->dres + (long)p*DIM, G->rms1[l], A->dn + (long)p*DIM,
                         A->res[l] + (long)p*DIM, P->rms1[l], DIM);
    }

    for (int p = 0; p < BT; p++) {                    /* embeddings */
        const float *d = A->dres + (long)p*DIM;
        float *gt = G->tok + (long)ix[p] * DIM;
        float *gp = G->pos + (long)(p % T) * DIM;
        for (int i = 0; i < DIM; i++) { gt[i] += d[i]; gp[i] += d[i]; }
    }
}

/* ------------------------------ AdamW ------------------------------- */
static void adamw(float *p, float *g, float *m, float *v, long n, int t) {
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float c1 = 1.0f - powf(b1, (float)t), c2 = 1.0f - powf(b2, (float)t);
    for (long i = 0; i < n; i++) {
        m[i] = b1 * m[i] + (1.0f - b1) * g[i];
        v[i] = b2 * v[i] + (1.0f - b2) * g[i] * g[i];
        float mh = m[i] / c1, vh = v[i] / c2;
        p[i] -= LEARN_RATE * (mh / (sqrtf(vh) + eps) + WEIGHT_DECAY * p[i]);
        g[i] = 0.0f;
    }
}

/* --------------------------- checkpoints ---------------------------- */
#define MAGIC 0x544C4C32   /* 'TLL2' */

static void save_ckpt(const char *path, const float *pb, const float *m,
                      const float *v, int step) {
#ifdef __wasi__
    FILE *f = fopen(path, "wb");        /* a-Shell: write in place */
#else
    char tmp[512];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    FILE *f = fopen(tmp, "wb");
#endif
    if (!f) { perror(path); return; }
    int hdr[8] = { MAGIC, DIM, NLAYER, NHEAD, CTX, FF, VS, step };
    long N = nparams();
    int ok = fwrite(hdr, sizeof hdr, 1, f) == 1
          && fwrite(dec_, 1, MAXV, f) == MAXV
          && fwrite(pb, sizeof(float), N, f) == (size_t)N
          && fwrite(m,  sizeof(float), N, f) == (size_t)N
          && fwrite(v,  sizeof(float), N, f) == (size_t)N;
    fclose(f);
#ifdef __wasi__
    if (!ok) fprintf(stderr, "checkpoint write failed\n");
#else
    if (ok) rename(tmp, path);
    else fprintf(stderr, "checkpoint write failed\n");
#endif
}

/* Reads config + charset, sets VS/enc/dec_. Returns -1 if file absent. */
static int load_header(const char *path, int *step) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    int hdr[8];
    if (fread(hdr, sizeof hdr, 1, f) != 1 || fread(dec_, 1, MAXV, f) != MAXV) {
        fclose(f); return -1;
    }
    fclose(f);
    if (hdr[0] != MAGIC || hdr[1] != DIM || hdr[2] != NLAYER ||
        hdr[3] != NHEAD || hdr[4] != CTX || hdr[5] != FF) {
        fprintf(stderr, "%s was made by a different config/version; "
                        "delete it and retrain\n", path);
        exit(1);
    }
    VS = hdr[6];
    memset(enc, 0, MAXV);
    for (int i = 0; i < VS; i++) enc[dec_[i]] = (unsigned char)i;
    if (step) *step = hdr[7];
    return 0;
}

static int load_ckpt(const char *path, float *pb, float *m, float *v) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    fseek(f, sizeof(int)*8 + MAXV, SEEK_SET);
    long N = nparams();
    if (fread(pb, sizeof(float), N, f) != (size_t)N) { fclose(f); return -1; }
    if (m && fread(m, sizeof(float), N, f) != (size_t)N) { fclose(f); return -1; }
    if (v && fread(v, sizeof(float), N, f) != (size_t)N) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

/* ---------------------------- sampling ------------------------------ */
static int sample_probs(const float *p, float temp) {
    float w[MAXV], s = 0.0f, it = 1.0f / temp;
    for (int i = 0; i < VS; i++) { w[i] = powf(p[i] + 1e-12f, it); s += w[i]; }
    float r = frand() * s, c = 0.0f;
    for (int i = 0; i < VS; i++) { c += w[i]; if (r <= c) return i; }
    return VS - 1;
}

static void sample_text(Params *P, Acts *A, const char *prompt, int n, float temp) {
    unsigned char win[CTX];
    int plen = (int)strlen(prompt);
    if (plen == 0) { prompt = "\n"; plen = 1; }
    int keep = plen > CTX ? CTX : plen;
    for (int i = 0; i < keep; i++)
        win[i] = enc[(unsigned char)prompt[plen - keep + i]];
    int len = keep;
    fwrite(prompt, 1, plen, stdout);
    for (int s = 0; s < n; s++) {
        forward(P, A, win, NULL, 1, len);
        int nx = sample_probs(A->logits + (long)(len - 1) * VS, temp);
        putchar(dec_[nx]);
        fflush(stdout);
        if (len == CTX) { memmove(win, win + 1, CTX - 1); win[CTX-1] = (unsigned char)nx; }
        else win[len++] = (unsigned char)nx;
    }
    putchar('\n');
}

/* ------------------------------ data -------------------------------- */
static unsigned char *read_file(const char *path, long *n) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    unsigned char *d = malloc(*n > 0 ? (size_t)*n : 1);
    if (!d || fread(d, 1, (size_t)*n, f) != (size_t)*n) {
        fprintf(stderr, "failed to read %s\n", path); exit(1);
    }
    fclose(f);
    return d;
}

static void encode_data(unsigned char *d, long n) {
    long unk = 0;
    for (long i = 0; i < n; i++) {
        unsigned char b = d[i];
        if (b != dec_[enc[b]]) unk++;   /* byte not in the charset */
        d[i] = enc[b];
    }
    if (unk) fprintf(stderr, "note: %ld bytes not in the model's charset "
                             "were mapped to id 0\n", unk);
}

/* --------------------- per-thread training worker ------------------- */
typedef struct {
    Params *P;
    Params G;
    float *gbuf;
    Acts A;
    const unsigned char *ix, *tg;
    float loss;
} Worker;

static void *work(void *arg) {
    Worker *w = (Worker *)arg;
    w->loss = forward(w->P, &w->A, w->ix, w->tg, BATCH/NTHREADS, CTX);
    backward(w->P, &w->G, &w->A, w->ix, w->tg, BATCH/NTHREADS, CTX);
    return NULL;
}

/* ------------------------------ modes ------------------------------- */
static void train(const char *datapath, const char *modelpath, int steps) {
    long dl;
    unsigned char *data = read_file(datapath, &dl);
    if (dl < CTX + 2) { fprintf(stderr, "need more data (%ld bytes)\n", dl); exit(1); }

    int step = 0;
    int resumed = (load_header(modelpath, &step) == 0);
    if (!resumed) build_charset(data, dl);
    encode_data(data, dl);

    long N = nparams();
    printf("tinyllm | %ld params | vocab %d, dim %d, %d layers, %d heads, "
           "ctx %d, batch %d, threads %d\n",
           N, VS, DIM, NLAYER, NHEAD, CTX, BATCH, NTHREADS);
    float *pb = xcalloc(N), *gb = xcalloc(N), *m = xcalloc(N), *v = xcalloc(N);
    if (resumed) {
        if (load_ckpt(modelpath, pb, m, v) != 0) {
            fprintf(stderr, "failed to read %s\n", modelpath); exit(1);
        }
        printf("resumed %s at step %d\n", modelpath, step);
    } else {
        init_params(pb);
        printf("initialized new model\n");
    }
    Params P; map_params(&P, pb);

    Worker W[NTHREADS];
    for (int t = 0; t < NTHREADS; t++) {
        W[t].P = &P;
        W[t].gbuf = xcalloc(N);
        map_params(&W[t].G, W[t].gbuf);
        alloc_acts(&W[t].A, BATCH/NTHREADS);
    }
    rngs ^= (unsigned long long)time(NULL) + (unsigned long long)step * 1000003ULL;

    unsigned char ix[BATCH*CTX], tg[BATCH*CTX];
    double lsum = 0.0; int lcnt = 0;
    double t0 = now_s();
    for (int s = 0; s < steps; s++) {
        for (int b = 0; b < BATCH; b++) {
            long off = (long)(xr() % (unsigned long long)(dl - CTX - 1));
            memcpy(ix + b*CTX, data + off, CTX);
            memcpy(tg + b*CTX, data + off + 1, CTX);
        }
        for (int t = 0; t < NTHREADS; t++) {
            W[t].ix = ix + (long)t * (BATCH/NTHREADS) * CTX;
            W[t].tg = tg + (long)t * (BATCH/NTHREADS) * CTX;
        }
#if NTHREADS > 1
        pthread_t th[NTHREADS];
        for (int t = 0; t < NTHREADS; t++) pthread_create(&th[t], NULL, work, &W[t]);
        for (int t = 0; t < NTHREADS; t++) pthread_join(th[t], NULL);
#else
        work(&W[0]);
#endif
        float loss = 0.0f;
        for (int t = 0; t < NTHREADS; t++) loss += W[t].loss;
        loss /= NTHREADS;
        for (long i = 0; i < N; i++) {              /* reduce worker grads */
            float sgr = 0.0f;
            for (int t = 0; t < NTHREADS; t++) { sgr += W[t].gbuf[i]; W[t].gbuf[i] = 0.0f; }
            gb[i] = sgr / NTHREADS;
        }
        step++;
        adamw(pb, gb, m, v, N, step);
        lsum += loss; lcnt++;
        if (step % 10 == 0 || s == steps - 1) {
            double secs = now_s() - t0;
            printf("step %6d | loss %.4f | %.0f tok/s\n",
                   step, lsum / lcnt, secs > 0 ? lcnt * (double)BATCH * CTX / secs : 0.0);
            fflush(stdout);
            lsum = 0.0; lcnt = 0; t0 = now_s();
        }
        if (step % CKPT_EVERY == 0) save_ckpt(modelpath, pb, m, v, step);
        if (step % SAMPLE_EVERY == 0) {
            printf("---- sample @ step %d ----\n", step);
            sample_text(&P, &W[0].A, "\n", 200, 0.8f);
            printf("--------------------------\n");
        }
    }
    save_ckpt(modelpath, pb, m, v, step);
    printf("saved %s at step %d\n", modelpath, step);
}

static void gen_cmd(const char *modelpath, const char *prompt, int n, float temp) {
    int step = 0;
    if (load_header(modelpath, &step) != 0) {
        fprintf(stderr, "can't open %s — train first\n", modelpath); exit(1);
    }
    long N = nparams();
    float *pb = xcalloc(N);
    if (load_ckpt(modelpath, pb, NULL, NULL) != 0) {
        fprintf(stderr, "failed to read %s\n", modelpath); exit(1);
    }
    Params P; map_params(&P, pb);
    Acts A; alloc_acts(&A, 1);
    rngs ^= (unsigned long long)time(NULL);
    sample_text(&P, &A, prompt, n, temp);
}

/* finite-difference check of the whole backward pass */
static void check(const char *datapath) {
    long dl;
    unsigned char *data = read_file(datapath, &dl);
    if (dl < CTX + 2) { fprintf(stderr, "need more data\n"); exit(1); }
    build_charset(data, dl);
    encode_data(data, dl);
    long N = nparams();
    float *pb = xcalloc(N), *gb = xcalloc(N);
    init_params(pb);
    Params P, G; map_params(&P, pb); map_params(&G, gb);
    Acts A; alloc_acts(&A, 1);
    unsigned char ix[CTX], tg[CTX];
    memcpy(ix, data, CTX); memcpy(tg, data + 1, CTX);
    float L0 = forward(&P, &A, ix, tg, 1, CTX);
    backward(&P, &G, &A, ix, tg, 1, CTX);

    /* spot checks (float32 + ReLU kinks make occasional outliers normal) */
    const float e = 3e-3f;
    printf("per-parameter spot checks (eps=%g):\n", e);
    for (int k = 0; k < 12; k++) {
        long i = (long)(xr() % (unsigned long long)N);
        float old = pb[i];
        pb[i] = old + e; float lp = forward(&P, &A, ix, tg, 1, CTX);
        pb[i] = old - e; float lm = forward(&P, &A, ix, tg, 1, CTX);
        pb[i] = old;
        float num = (lp - lm) / (2.0f * e), ana = gb[i];
        float rel = fabsf(num - ana) / (fabsf(num) + fabsf(ana) + 1e-8f);
        printf("param %8ld | analytic %+.6f | numeric %+.6f | rel %.4f\n", i, ana, num, rel);
    }

    /* decisive test: centered slope along grad direction must equal |grad|^2 */
    (void)L0;
    double g2 = 0.0;
    for (long i = 0; i < N; i++) g2 += (double)gb[i] * gb[i];
    float a = 0.01f / (float)sqrt(g2 > 1e-12 ? g2 : 1e-12);
    for (long i = 0; i < N; i++) pb[i] -= a * gb[i];
    float Lm = forward(&P, &A, ix, tg, 1, CTX);
    for (long i = 0; i < N; i++) pb[i] += 2.0f * a * gb[i];
    float Lp = forward(&P, &A, ix, tg, 1, CTX);
    for (long i = 0; i < N; i++) pb[i] -= a * gb[i];
    double slope = (double)(Lp - Lm) / (2.0 * a);
    double rel = fabs(slope - g2) / g2;
    printf("directional test: |grad|^2 %.6f vs measured slope %.6f (rel %.4f)\n",
           g2, slope, rel);
    if (rel < 0.03) printf("OK: backward pass matches finite differences\n");
    else { printf("FAIL: backward pass looks wrong\n"); exit(1); }
}

/* ------------------------------ main -------------------------------- */
int main(int argc, char **argv) {
    if (argc >= 4 && !strcmp(argv[1], "train")) {
        train(argv[2], argv[3], argc > 4 ? atoi(argv[4]) : 1000);
    } else if (argc >= 3 && !strcmp(argv[1], "gen")) {
        gen_cmd(argv[2], argc > 3 ? argv[3] : "\n",
                argc > 4 ? atoi(argv[4]) : 400,
                argc > 5 ? (float)atof(argv[5]) : 0.8f);
    } else if (argc >= 3 && !strcmp(argv[1], "check")) {
        check(argv[2]);
    } else {
        fprintf(stderr,
            "usage:\n"
            "  %s train <data.txt> <model.bin> [steps=1000]\n"
            "  %s gen   <model.bin> [prompt] [ntokens=400] [temp=0.8]\n"
            "  %s check <data.txt>\n", argv[0], argv[0], argv[0]);
        return 1;
    }
    return 0;
}
