"""
island_net.py — Island Memory Network (core)
=============================================
A continual-learning memory field built on the one piece of the RH blind-zone
mathematics that transfers literally: the INFLUENCE KERNEL of a pole.

The field:   F(s) = sum_k  C_k / (s - rho_k),     s, rho_k in C,  C_k in C^D

  - "Working memory" reads the field on the boundary line Re(s) = 0:
    an input x is projected to J positions t_j(x) on the line and the
    features are Re/Im of F(i t_j(x)).
  - A memory slot at depth d = Re(rho_k) contributes to the boundary
    readout with magnitude  ~ |C_k|/d          (kernel 1/(s-rho))
    and receives boundary-loss gradients
        dL/dC_k   ~ 1/d        (adjoint of the kernel)
        dL/drho_k ~ 1/d**2     (kernel derivative C/(s-rho)^2)
  - ARCHIVAL = translating a task's slots to depth d_cold. Nothing is
    frozen and no gating exists: interference is suppressed by geometry
    alone, with exponents the code lets you verify.
  - RETRIEVAL = shifting the readout contour to the island's depth
    (s = d_cold + i t), where the archived slots dominate and the hot
    line is suppressed by the same kernel — or "thawing" (translating
    the slots back), which is exact because archival is a translation.

Honest framing: freezing old parameters ALSO gives zero interference and is
simpler. What the field buys over freezing: (1) interference is a continuous
dial (graded warm/cold), not a binary mask; (2) archived content keeps
contributing (faintly, ~1/d) and remains readable in superposition by contour
shift; (3) the suppression law is analytic and checkable. The experiments
include the freezing control. Pure numpy, analytic gradients — the 1/d^2 law
is literally visible in the backward pass.

PerceptionLab / Antti Luode, with Claude. Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class SlotBlock:
    """One task's memory slots: complex anchors rho (K,) and patterns C (K,D)."""
    def __init__(self, K, D, t_spread, sigma_hot, rng):
        self.rho = (sigma_hot + 0.05 * rng.standard_normal(K)
                    + 1j * rng.uniform(-t_spread, t_spread, K))
        self.C = 0.3 * (rng.standard_normal((K, D)) + 1j * rng.standard_normal((K, D)))
        self.frozen = False

    def translate(self, d_sigma=0.0, d_t=0.0):
        """Archival / thaw is a pure translation — lossless by construction."""
        self.rho = self.rho + d_sigma + 1j * d_t


class IslandNet:
    """Cauchy-kernel memory field + per-task linear heads."""
    def __init__(self, in_dim=16, D=8, J=8, t_spread=8.0, sigma_hot=0.35, seed=0):
        self.rng = np.random.default_rng(seed)
        self.D, self.J = D, J
        self.Wt = self.rng.standard_normal((in_dim, J)) / np.sqrt(in_dim) * t_spread / 2
        self.t_spread, self.sigma_hot = t_spread, sigma_hot
        self.blocks = {}      # task_id -> SlotBlock
        self.heads = {}       # task_id -> (W, b)

    # ---------------- field machinery ----------------
    def _gather(self):
        rhos, Cs, owners = [], [], []
        for tid, blk in self.blocks.items():
            rhos.append(blk.rho); Cs.append(blk.C)
            owners += [tid] * len(blk.rho)
        return np.concatenate(rhos), np.concatenate(Cs), owners

    def features(self, X, contour_sigma=0.0, contour_t=0.0):
        """Re/Im of F(s) at s = contour + i t_j(x). Returns feat, cache."""
        rho, C, owners = self._gather()
        t = X @ self.Wt                                   # (B,J)
        s = contour_sigma + 1j * (t + contour_t)
        Kk = 1.0 / (s[:, :, None] - rho[None, None, :])   # (B,J,K) complex
        F = Kk @ C                                        # (B,J,D) complex
        feat = np.concatenate([F.real, F.imag], -1).reshape(len(X), -1)
        return feat, (Kk, C, rho, owners, s)

    def logits(self, X, task_id, **kw):
        feat, cache = self.features(X, **kw)
        W, b = self.heads[task_id]
        return feat @ W + b, feat, cache

    # ---------------- training ----------------
    def new_task(self, task_id, n_classes, K=24):
        self.blocks[task_id] = SlotBlock(K, self.D, self.t_spread, self.sigma_hot, self.rng)
        W = 0.01 * self.rng.standard_normal((2 * self.D * self.J, n_classes))
        self.heads[task_id] = [W, np.zeros(n_classes)]

    def train_task(self, task_id, X, y, steps=700, bs=64,
                   lr_C=0.04, lr_rho=0.015, lr_W=0.08, train_heads_only=False):
        """SGD with analytic gradients. Gradients flow to ALL non-frozen blocks
        (that is the interference path the geometry must suppress)."""
        n_classes = self.heads[task_id][0].shape[1]
        for step in range(steps):
            idx = self.rng.integers(0, len(X), bs)
            xb, yb = X[idx], y[idx]
            logit, feat, (Kk, C, rho, owners, s) = self.logits(xb, task_id)
            p = softmax(logit)
            G = (p - np.eye(n_classes)[yb]) / bs          # (B,nc)
            W, b = self.heads[task_id]
            dW = feat.T @ G
            dfeat = G @ W.T                               # (B, 2JD)
            B = len(xb)
            dF = dfeat.reshape(B, self.J, 2 * self.D)
            U = dF[..., :self.D] + 1j * dF[..., self.D:]  # complex adjoint of F
            # dL/dC_k = sum_bj conj(K)_bjk * U_bjd        (~ 1/d   for deep slots)
            dC = np.einsum('bjk,bjd->kd', np.conj(Kk), U)
            # dL/drho: W_k = sum conj(U) C K^2            (~ 1/d^2 for deep slots)
            Wk = np.einsum('bjd,kd,bjk->k', np.conj(U), C, Kk ** 2)
            drho = Wk.real - 1j * Wk.imag                 # d/d(rho_r), -d/d(rho_i)
            # scatter to blocks
            self.heads[task_id][0] -= lr_W * dW
            self.heads[task_id][1] -= lr_W * G.sum(0)
            if not train_heads_only:
                i0 = 0
                for tid, blk in self.blocks.items():
                    k = len(blk.rho); sl = slice(i0, i0 + k); i0 += k
                    if blk.frozen:
                        continue
                    blk.C -= lr_C * np.conj(dC[sl])
                    blk.rho -= lr_rho * np.conj(drho[sl])
                    # keep anchors off the readout contour (numerical guard)
                    blk.rho = np.where(blk.rho.real < 0.1,
                                       0.1 + 1j * blk.rho.imag, blk.rho)

    def accuracy(self, X, y, task_id, **kw):
        logit, _, _ = self.logits(X, task_id, **kw)
        return float((logit.argmax(1) == y).mean())

    # ---------------- island ops ----------------
    def archive(self, task_id, depth):
        self.blocks[task_id].translate(d_sigma=depth)

    def thaw(self, task_id, depth):
        self.blocks[task_id].translate(d_sigma=-depth)

    def inject(self, task_id, block, head):
        """Concept injection: drop a pre-trained island into the field."""
        self.blocks[task_id] = block
        self.heads[task_id] = head
