#!/usr/bin/env python3
"""tinyllm in MLX - the same byte-level GPT, running on the Apple GPU.

One file, two backends. It imports Apple's MLX if present and falls back to
numpy otherwise, and the model itself is written only in array operations
that both provide. That is deliberate: the forward pass you run on the GPU
is character-for-character the same code verified against numpy on the CPU.

    pip install mlx            # ~50 MB, no Xcode needed, Apple Silicon only

    python3 tinyllm_mlx.py check                       # verify the math
    python3 tinyllm_mlx.py train snake_train.txt 2000
    python3 tinyllm_mlx.py gen "def " 400 0.5
    python3 tinyllm_mlx.py bench                       # tok/s, compare to C

Sizes are set by flags, e.g. --dim 256 --layers 4 --ctx 128 --batch 16.
Checkpoints are .npz and are NOT interchangeable with the C version's
model.bin (different weight layout); train and sample within one tool.
"""
import argparse
import math
import os
import sys
import time

import numpy as np

# --------------------------------------------------------------------------
# backend: MLX on Apple Silicon, numpy everywhere else
# --------------------------------------------------------------------------
try:
    import mlx.core as xp
    BACKEND = "mlx"
except ImportError:
    xp = np
    BACKEND = "numpy"


def to_backend(a):
    """numpy array -> backend array."""
    return xp.array(a) if BACKEND == "mlx" else np.asarray(a, dtype=np.float32)


def to_numpy(a):
    return np.array(a, copy=True) if BACKEND == "mlx" else np.asarray(a)


def realize(*things):
    """MLX is lazy; force evaluation. No-op on numpy."""
    if BACKEND == "mlx":
        xp.eval(*things)


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
class Config:
    def __init__(self, dim=64, layers=2, heads=4, ctx=128, vocab=82):
        self.dim, self.layers, self.heads, self.ctx, self.vocab = \
            dim, layers, heads, ctx, vocab
        self.ff = 4 * dim
        assert dim % heads == 0, "dim must divide evenly by heads"
        self.hd = dim // heads


def init_params(cfg, seed=0):
    """Weights live as [in, out] here (x @ W), unlike the C version's [out, in]."""
    rng = np.random.default_rng(seed)
    s = 0.02
    resid = s / math.sqrt(2 * cfg.layers)     # scaled residual init
    P = {
        "tok": rng.normal(0, s, (cfg.vocab, cfg.dim)).astype(np.float32),
        "pos": rng.normal(0, 0.01, (cfg.ctx, cfg.dim)).astype(np.float32),
        "rmsf": np.ones(cfg.dim, np.float32),
        "head": rng.normal(0, s, (cfg.dim, cfg.vocab)).astype(np.float32),
    }
    for l in range(cfg.layers):
        P["rms1_%d" % l] = np.ones(cfg.dim, np.float32)
        P["wqkv_%d" % l] = rng.normal(0, s, (cfg.dim, 3 * cfg.dim)).astype(np.float32)
        P["wo_%d" % l] = rng.normal(0, resid, (cfg.dim, cfg.dim)).astype(np.float32)
        P["rms2_%d" % l] = np.ones(cfg.dim, np.float32)
        P["w1_%d" % l] = rng.normal(0, s, (cfg.dim, cfg.ff)).astype(np.float32)
        P["w2_%d" % l] = rng.normal(0, resid, (cfg.ff, cfg.dim)).astype(np.float32)
    return {k: to_backend(v) for k, v in P.items()}


def rmsnorm(x, w, eps=1e-5):
    ms = xp.mean(x * x, axis=-1, keepdims=True)
    return x * (1.0 / xp.sqrt(ms + eps)) * w


def softmax_last(s):
    m = xp.max(s, axis=-1, keepdims=True)
    e = xp.exp(s - m)
    return e / xp.sum(e, axis=-1, keepdims=True)


def causal_mask(T):
    """Additive mask, precomputed as a constant so no boolean ops enter the graph."""
    m = np.zeros((T, T), np.float32)
    m[np.triu_indices(T, k=1)] = -1e9
    return to_backend(m)


def forward(P, cfg, idx, mask):
    """idx: (B, T) int token ids. Returns logits (B, T, vocab)."""
    B, T = idx.shape
    H, HD = cfg.heads, cfg.hd
    scale = 1.0 / math.sqrt(HD)

    x = xp.take(P["tok"], idx, axis=0) + P["pos"][:T]          # (B,T,D)

    for l in range(cfg.layers):
        h = rmsnorm(x, P["rms1_%d" % l])
        qkv = xp.matmul(h, P["wqkv_%d" % l])                   # (B,T,3D)
        q = qkv[:, :, 0 * cfg.dim:1 * cfg.dim]
        k = qkv[:, :, 1 * cfg.dim:2 * cfg.dim]
        v = qkv[:, :, 2 * cfg.dim:3 * cfg.dim]
        # (B,T,D) -> (B,H,T,HD)
        q = xp.transpose(xp.reshape(q, (B, T, H, HD)), (0, 2, 1, 3))
        k = xp.transpose(xp.reshape(k, (B, T, H, HD)), (0, 2, 1, 3))
        v = xp.transpose(xp.reshape(v, (B, T, H, HD)), (0, 2, 1, 3))

        att = xp.matmul(q, xp.transpose(k, (0, 1, 3, 2))) * scale
        att = softmax_last(att + mask[:T, :T])
        y = xp.matmul(att, v)                                  # (B,H,T,HD)
        y = xp.reshape(xp.transpose(y, (0, 2, 1, 3)), (B, T, cfg.dim))
        x = x + xp.matmul(y, P["wo_%d" % l])

        h2 = rmsnorm(x, P["rms2_%d" % l])
        hidden = xp.maximum(xp.matmul(h2, P["w1_%d" % l]), 0.0)   # ReLU
        x = x + xp.matmul(hidden, P["w2_%d" % l])

    x = rmsnorm(x, P["rmsf"])
    return xp.matmul(x, P["head"])


def loss_fn(P, cfg, idx, tgt_onehot, mask):
    """Mean cross-entropy. Targets arrive one-hot to avoid backend-specific
    advanced indexing (vocab is small, so the cost is negligible)."""
    logits = forward(P, cfg, idx, mask)
    m = xp.max(logits, axis=-1, keepdims=True)
    z = logits - m
    logZ = xp.log(xp.sum(xp.exp(z), axis=-1, keepdims=True))
    logp = z - logZ
    return -xp.mean(xp.sum(logp * tgt_onehot, axis=-1))


def onehot(tgt, vocab):
    B, T = tgt.shape
    out = np.zeros((B, T, vocab), np.float32)
    out[np.arange(B)[:, None], np.arange(T)[None, :], tgt] = 1.0
    return to_backend(out)


# --------------------------------------------------------------------------
# AdamW, written in plain array ops so it runs on either backend
# --------------------------------------------------------------------------
class AdamW:
    def __init__(self, P, lr=1e-3, wd=1e-4, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.wd, self.b1, self.b2, self.eps = lr, wd, b1, b2, eps
        self.t = 0
        self.m = {k: to_backend(np.zeros(to_numpy(v).shape, np.float32)) for k, v in P.items()}
        self.v = {k: to_backend(np.zeros(to_numpy(v).shape, np.float32)) for k, v in P.items()}

    def step(self, P, G):
        self.t += 1
        c1 = 1.0 - self.b1 ** self.t
        c2 = 1.0 - self.b2 ** self.t
        for k in P:
            g = G[k]
            self.m[k] = self.b1 * self.m[k] + (1.0 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1.0 - self.b2) * g * g
            mh = self.m[k] / c1
            vh = self.v[k] / c2
            P[k] = P[k] - self.lr * (mh / (xp.sqrt(vh) + self.eps) + self.wd * P[k])
        return P


def make_grad_fn(cfg, mask):
    """MLX gives us autodiff; numpy has none, so it only runs forward."""
    if BACKEND == "mlx":
        f = lambda P, idx, oh: loss_fn(P, cfg, idx, oh, mask)
        return xp.value_and_grad(f)
    def forward_only(P, idx, oh):
        return loss_fn(P, cfg, idx, oh, mask), None
    return forward_only


# --------------------------------------------------------------------------
# data / charset
# --------------------------------------------------------------------------
def load_corpus(path):
    raw = open(path, "rb").read()
    chars = sorted(set(raw))
    enc = {b: i for i, b in enumerate(chars)}
    data = np.array([enc[b] for b in raw], dtype=np.int32)
    return data, bytes(chars)


def get_batch(data, batch, ctx, rng):
    off = rng.integers(0, len(data) - ctx - 1, size=batch)
    idx = np.stack([data[o:o + ctx] for o in off]).astype(np.int32)
    tgt = np.stack([data[o + 1:o + ctx + 1] for o in off]).astype(np.int32)
    return idx, tgt


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_train(args):
    data, charset = load_corpus(args.corpus)
    cfg = Config(args.dim, args.layers, args.heads, args.ctx, len(charset))
    mask = causal_mask(cfg.ctx)
    P = init_params(cfg, args.seed)
    start = 0
    if args.resume and os.path.exists(args.model):
        P, start, charset, cfg = load_ckpt(args.model)
        print("resumed %s at step %d" % (args.model, start))
    nparams = sum(int(np.prod(to_numpy(v).shape)) for v in P.values())
    print("backend %s | %d params | dim %d, %d layers, ctx %d, batch %d, vocab %d"
          % (BACKEND, nparams, cfg.dim, cfg.layers, cfg.ctx, args.batch, cfg.vocab))
    if BACKEND == "numpy":
        print("WARNING: numpy backend has no autodiff - training needs MLX")
        return

    grad_fn = make_grad_fn(cfg, mask)
    opt = AdamW(P, lr=args.lr)
    rng = np.random.default_rng(args.seed + 1)
    t0 = time.time()
    run, n = 0.0, 0
    for step in range(start + 1, start + args.steps + 1):
        idx, tgt = get_batch(data, args.batch, cfg.ctx, rng)
        loss, G = grad_fn(P, to_backend_int(idx), onehot(tgt, cfg.vocab))
        P = opt.step(P, G)
        realize(*P.values())
        run += float(loss); n += 1
        if step % 10 == 0:
            dt = time.time() - t0
            print("step %6d | loss %.4f | %.0f tok/s"
                  % (step, run / n, n * args.batch * cfg.ctx / dt))
            run, n, t0 = 0.0, 0, time.time()
        if step % 200 == 0:
            save_ckpt(args.model, P, step, charset, cfg)
    save_ckpt(args.model, P, start + args.steps, charset, cfg)
    print("saved", args.model)


def to_backend_int(a):
    return xp.array(a) if BACKEND == "mlx" else np.asarray(a)


def save_ckpt(path, P, step, charset, cfg):
    out = {k: to_numpy(v) for k, v in P.items()}
    out["__step"] = np.array([step])
    out["__charset"] = np.frombuffer(charset, dtype=np.uint8)
    out["__cfg"] = np.array([cfg.dim, cfg.layers, cfg.heads, cfg.ctx, cfg.vocab])
    np.savez(path, **out)


def load_ckpt(path, cfg=None):
    z = np.load(path if path.endswith(".npz") else path + ".npz")
    step = int(z["__step"][0])
    charset = bytes(z["__charset"].tolist())
    dim, layers, heads, ctx, vocab = [int(x) for x in z["__cfg"]]
    cfg = Config(dim, layers, heads, ctx, vocab)
    P = {k: to_backend(z[k]) for k in z.files if not k.startswith("__")}
    return P, step, charset, cfg


def cmd_gen(args):
    P, step, charset, cfg = load_ckpt(args.model)
    mask = causal_mask(cfg.ctx)
    enc = {b: i for i, b in enumerate(charset)}
    prompt = args.prompt.encode()
    ids = [enc.get(b, 0) for b in prompt]
    sys.stdout.write(args.prompt)
    rng = np.random.default_rng()
    for _ in range(args.n):
        window = ids[-cfg.ctx:]
        idx = np.array([window], dtype=np.int32)
        logits = to_numpy(forward(P, cfg, to_backend_int(idx), mask))[0, len(window) - 1]
        logits = logits / max(args.temp, 1e-6)
        p = np.exp(logits - logits.max())
        p /= p.sum()
        nxt = int(rng.choice(len(p), p=p))
        ids.append(nxt)
        sys.stdout.write(chr(charset[nxt]))
        sys.stdout.flush()
    print()


def cmd_bench(args):
    cfg = Config(args.dim, args.layers, args.heads, args.ctx, 82)
    mask = causal_mask(cfg.ctx)
    P = init_params(cfg, 0)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, cfg.vocab, (args.batch, cfg.ctx)).astype(np.int32)
    tgt = rng.integers(0, cfg.vocab, (args.batch, cfg.ctx)).astype(np.int32)
    oh = onehot(tgt, cfg.vocab)
    bidx = to_backend_int(idx)
    grad_fn = make_grad_fn(cfg, mask)
    for _ in range(3):                       # warm up / compile
        out = grad_fn(P, bidx, oh)
        realize(out[0])
    t0 = time.time()
    steps = args.steps
    for _ in range(steps):
        loss, G = grad_fn(P, bidx, oh)
        realize(loss)
    dt = time.time() - t0
    print("backend %s | dim %d layers %d ctx %d batch %d" %
          (BACKEND, cfg.dim, cfg.layers, cfg.ctx, args.batch))
    print("%.1f ms/step | %.0f tok/s%s"
          % (1000 * dt / steps, steps * args.batch * cfg.ctx / dt,
             "" if BACKEND == "mlx" else "   [FORWARD ONLY - numpy has no"
                                         " autodiff; not comparable to C]"))


def cmd_check(args):
    """Backend-independent correctness tests for the model definition."""
    cfg = Config(16, 2, 2, 8, 11)
    mask = causal_mask(cfg.ctx)
    P = init_params(cfg, 0)
    rng = np.random.default_rng(3)
    idx = rng.integers(0, cfg.vocab, (2, cfg.ctx)).astype(np.int32)
    tgt = rng.integers(0, cfg.vocab, (2, cfg.ctx)).astype(np.int32)
    oh = onehot(tgt, cfg.vocab)

    L = float(loss_fn(P, cfg, to_backend_int(idx), oh, mask))
    expect = math.log(cfg.vocab)
    print("1. loss at init      %.4f (expect ~%.4f = ln %d)" % (L, expect, cfg.vocab))
    assert abs(L - expect) < 0.35, "initial loss is wrong"

    logits = to_numpy(forward(P, cfg, to_backend_int(idx), mask))
    bumped = idx.copy()
    t = 5
    bumped[0, t] = (bumped[0, t] + 1) % cfg.vocab
    logits2 = to_numpy(forward(P, cfg, to_backend_int(bumped), mask))
    before = np.abs(logits[0, :t] - logits2[0, :t]).max()
    after = np.abs(logits[0, t:] - logits2[0, t:]).max()
    print("2. causality         positions <t change by %.2e, >=t by %.2e" % (before, after))
    assert before < 1e-6, "future token leaked into the past - mask is broken"
    assert after > 1e-6, "changing a token had no effect at all"

    shp = to_numpy(forward(P, cfg, to_backend_int(idx), mask)).shape
    print("3. logits shape      %s" % (shp,))
    assert shp == (2, cfg.ctx, cfg.vocab)

    print("4. gradient descent  (finite differences on a tiny model)")
    tiny = Config(4, 1, 1, 4, 5)
    tmask = causal_mask(tiny.ctx)
    TP = init_params(tiny, 1)
    ti = rng.integers(0, tiny.vocab, (1, tiny.ctx)).astype(np.int32)
    tt = rng.integers(0, tiny.vocab, (1, tiny.ctx)).astype(np.int32)
    toh = onehot(tt, tiny.vocab)
    bti = to_backend_int(ti)
    f = lambda Q: float(loss_fn(Q, tiny, bti, toh, tmask))
    n_par = sum(int(np.prod(to_numpy(v).shape)) for v in TP.values())
    losses = []
    for it in range(12):
        base = f(TP)
        losses.append(base)
        eps, lr = 1e-3, 0.5
        for k in TP:                              # full finite-difference grad
            flat = to_numpy(TP[k]).ravel().copy()
            g = np.zeros_like(flat)
            for i in range(flat.size):
                keep = flat[i]
                flat[i] = keep + eps
                TP[k] = to_backend(flat.reshape(to_numpy(TP[k]).shape))
                lp = f(TP)
                flat[i] = keep - eps
                TP[k] = to_backend(flat.reshape(to_numpy(TP[k]).shape))
                lm = f(TP)
                flat[i] = keep
                g[i] = (lp - lm) / (2 * eps)
            TP[k] = to_backend((flat - lr * g).reshape(to_numpy(TP[k]).shape))
    print("   %d params, loss %.4f -> %.4f over %d steps"
          % (n_par, losses[0], losses[-1], len(losses)))
    assert losses[-1] < losses[0] - 0.05, "finite-difference training did not learn"
    print("\nOK: model definition is correct on the %s backend" % BACKEND)


def main():
    ap = argparse.ArgumentParser(description="tinyllm on MLX / numpy")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--dim", type=int, default=64)
        p.add_argument("--layers", type=int, default=2)
        p.add_argument("--heads", type=int, default=4)
        p.add_argument("--ctx", type=int, default=128)
        p.add_argument("--batch", type=int, default=16)

    t = sub.add_parser("train"); common(t)
    t.add_argument("corpus"); t.add_argument("steps", type=int, nargs="?", default=2000)
    t.add_argument("--model", default="model_mlx.npz")
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", action="store_true")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("gen")
    g.add_argument("prompt"); g.add_argument("n", type=int, nargs="?", default=400)
    g.add_argument("temp", type=float, nargs="?", default=0.5)
    g.add_argument("--model", default="model_mlx.npz")
    g.set_defaults(func=cmd_gen)

    b = sub.add_parser("bench"); common(b)
    b.add_argument("steps", type=int, nargs="?", default=30)
    b.set_defaults(func=cmd_bench)

    c = sub.add_parser("check")
    c.set_defaults(func=cmd_check)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
