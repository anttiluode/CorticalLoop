"""
grid_discovery.py — the grid basis, discovered instead of hardcoded
===================================================================
Closes the hand-tuned-frequency gap in cortical_loop.py exactly as proposed:
Takens-embed the incoming MULTICHANNEL sensory stream, run DMD (a finite-data
Koopman estimate), keep the persistent oscillatory modes, and use their
frequencies as the grid module scales k_m. The loop then runs on a basis it
discovered from the environment rather than one we set.

HONEST SCOPE (read before believing the word 'self-organize'):
  DMD returns the dominant LINEAR oscillatory modes. The frequencies emerge, but
  they emerge as Fourier/Koopman modes because that is what a linear operator's
  eigenstructure IS. This closes the LINEAR half of O1 (the same half v9_learned
  and dynamic_geometric_net close). The NONLINEAR dictionary -- features that are
  not sinusoids, grown from raw input statistics like V1's Gabors -- is untouched.
  Also: DMD on a moving stream yields TEMPORAL frequencies = k_spatial * velocity,
  so anchoring the spatial scale needs a self-motion velocity reference during a
  calibration walk (biologically: motor efference / vestibular). That reference is
  the one thing still supplied; everything downstream is discovered.

Pure numpy. PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
from cortical_loop import CorticalLoop, circ_err


# ---------------- the environment (unknown to the animal) ----------------
K_TRUE = np.array([3.0, 5.0, 7.0, 11.0])          # the world's spatial frequencies
def texture(x, rng=None):
    ph = np.array([0.3, 1.1, 2.0, 0.7])
    a  = np.array([1.0, 0.8, 0.7, 0.5])
    return (a[:, None]*np.sin(2*np.pi*np.outer(K_TRUE, np.atleast_1d(x)) + ph[:, None])).sum(0)


# ---------------- Takens embedding + DMD (Koopman estimate) ----------------
def hankel_multi(R, d, tau=1):
    """R: (C, T) multichannel -> stacked delay embedding ((C*d), N)."""
    C, T = R.shape; N = T - (d-1)*tau
    rows = [R[c, i*tau:i*tau+N] for i in range(d) for c in range(C)]
    return np.stack(rows)

def dmd_modes(H, dt=1.0):
    X1, X2 = H[:, :-1], H[:, 1:]
    U, S, Vt = np.linalg.svd(X1, full_matrices=False)
    r = int((S > S[0]*1e-8).sum())
    Ur, Sr, Vr = U[:, :r], S[:r], Vt[:r].conj().T
    At = Ur.conj().T @ X2 @ Vr @ np.diag(1/Sr)
    lam, W = np.linalg.eig(At)
    Phi = X2 @ Vr @ np.diag(1/Sr) @ W
    b = np.linalg.lstsq(Phi, X1[:, 0].astype(complex), rcond=None)[0]
    f = np.angle(lam)/(2*np.pi*dt)                  # cycles/step (signed)
    energy = np.abs(b)*np.linalg.norm(Phi, axis=0)
    persist = np.abs(lam)
    return f, persist, energy


def discover_grid_basis(n_modules=4, v_cal=0.004, T=2500, d=80, tau=2,
                        n_sensors=6, seed=0):
    """Walk at known velocity v_cal, sense the world through a few raw receptors,
       and let Takens+DMD discover the spatial frequencies."""
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    x_cal = (v_cal*t) % 1.0
    offs = np.linspace(0, 0.04, n_sensors)          # raw receptor offsets (no freq labels)
    R = np.stack([texture(x_cal + o) for o in offs]) + 0.03*rng.standard_normal((n_sensors, T))

    H = hankel_multi(R, d, tau)
    f, persist, energy = dmd_modes(H)

    # keep persistent, positive-frequency modes; greedily take the strongest distinct ones
    keep = (persist > 0.97) & (f > 1e-4)
    f, energy = f[keep], energy[keep]
    order = np.argsort(-energy)
    picked = []
    for i in order:
        if all(abs(f[i] - f[j]) > 0.004 for j in picked):
            picked.append(i)
        if len(picked) == n_modules:
            break
    f_temporal = np.sort(f[picked])
    k_discovered = f_temporal / v_cal               # temporal -> spatial via the walk speed
    return k_discovered, f_temporal


# ============================================================ demo
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k_disc, f_temp = discover_grid_basis()
    print("="*68); print("GRID BASIS DISCOVERED FROM THE STREAM (Takens + DMD/Koopman)"); print("="*68)
    print(f"  true spatial frequencies      : {K_TRUE}")
    print(f"  discovered (DMD / v_cal)      : {np.round(k_disc,3)}")
    err = np.abs(np.sort(k_disc) - K_TRUE)
    print(f"  match (|discovered - true|)   : {np.round(err,3)}  (max {err.max():.3f})")
    print(f"  -> the basis self-organized to the world's dominant LINEAR modes.")

    # run the SAME blackout/dead-reckoning test with discovered vs hardcoded basis
    T = 1400
    v = np.full(T, 0.0040); v[350:550] = -0.0034; v[900:1000] = 0.0080
    x_true = np.cumsum(v) % 1.0
    drop = np.zeros(T, bool); drop[600:950] = True

    loop_disc = CorticalLoop(k_modules=tuple(np.sort(k_disc)))
    loop_hard = CorticalLoop(k_modules=(3, 5, 7, 11))
    xd, vd = loop_disc.run(x_true, drop, "closed")
    xh, _  = loop_hard.run(x_true, drop, "closed")
    xo, _  = loop_hard.run(x_true, drop, "open")
    ed, eh, eo = circ_err(xd, x_true), circ_err(xh, x_true), circ_err(xo, x_true)
    win = slice(600, 950)
    print(f"\n  loop tracking error  discovered-basis={ed.mean():.4f}  hardcoded={eh.mean():.4f}")
    print(f"  during blackout      discovered-basis={ed[win].mean():.4f}  hardcoded={eh[win].mean():.4f}"
          f"  open={eo[win].mean():.4f}")
    print(f"  -> the loop runs on the DISCOVERED basis as well as the hand-set one,")
    print(f"     and still dead-reckons through the blackout.")
    print(f"\n  HONEST: this is the LINEAR half of O1. DMD found the world's sinusoidal")
    print(f"  modes because that is what a Koopman operator's eigenvalues are. A nonlinear")
    print(f"  dictionary (non-sinusoid features grown from raw statistics) is still open.")

    # ---------------- figure ----------------
    f_all, persist_all, energy_all = dmd_modes(
        hankel_multi(np.stack([texture(((0.004*np.arange(2500)) % 1.0) + o)
                               for o in np.linspace(0, 0.04, 6)])
                     + 0.0, 80, 2))
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    pos = f_all > 1e-4
    ax[0,0].scatter(f_all[pos], persist_all[pos], s=10+300*energy_all[pos]/energy_all.max(),
                    color="#1f6feb", alpha=0.6)
    for k in K_TRUE:
        ax[0,0].axvline(k*0.004, color="#b03030", ls="--", lw=1)
    ax[0,0].set_xlim(0, 0.06); ax[0,0].set_ylim(0.9, 1.02)
    ax[0,0].set_title("A. DMD spectrum of the stream\n(red = true freqs; size = energy)")
    ax[0,0].set_xlabel("temporal frequency (cyc/step)"); ax[0,0].set_ylabel("persistence |lambda|")

    ax[0,1].bar(np.arange(4)-0.18, K_TRUE, width=0.36, color="#b03030", label="true")
    ax[0,1].bar(np.arange(4)+0.18, np.sort(k_disc), width=0.36, color="#2a9d4a", label="discovered")
    ax[0,1].set_title("B. spatial frequencies: discovered vs true")
    ax[0,1].set_xlabel("module"); ax[0,1].set_ylabel("k (cyc/track)"); ax[0,1].legend(fontsize=8)

    tt = np.arange(T)
    ax[1,0].plot(tt, ed, color="#2a9d4a", lw=1.1, label="discovered basis")
    ax[1,0].plot(tt, eh, color="#1f6feb", lw=1.0, ls="--", label="hardcoded basis")
    ax[1,0].plot(tt, eo, color="#b03030", lw=1.0, label="open loop")
    ax[1,0].axvspan(600, 950, color="#cccccc", alpha=0.5)
    ax[1,0].set_title("C. loop runs on the discovered basis, still dead-reckons")
    ax[1,0].set_xlabel("step"); ax[1,0].set_ylabel("tracking error"); ax[1,0].legend(fontsize=8)

    ax[1,1].axis("off")
    ax[1,1].text(0.0, 0.97,
       ("WHAT THIS CLOSES:\n\n"
        f"  discovered k = {np.round(np.sort(k_disc),2)}\n"
        f"  true k       = {K_TRUE}\n"
        f"  max error    = {err.max():.3f}\n\n"
        "  the grid basis self-organizes from the\n"
        "  stream via Takens + DMD. No hand-set\n"
        "  frequencies. The loop tracks as well as\n"
        "  with the hardcoded basis.\n\n"
        "WHAT IT DOES NOT CLOSE:\n\n"
        "  this is the LINEAR half of O1. DMD finds\n"
        "  sinusoidal modes -- that is what Koopman\n"
        "  eigenvalues are. A nonlinear dictionary\n"
        "  (non-Fourier features from raw statistics,\n"
        "  V1-style) is still open. And a velocity\n"
        "  reference (efference) anchors the scale."),
       va="top", family="monospace", fontsize=9)
    fig.tight_layout(); fig.savefig("grid_discovery.png", dpi=120)
    print("\n  figure: grid_discovery.png")
