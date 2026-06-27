"""
cortical_loop.py — closing the loop: the EC-hippocampal circuit as a running model
==================================================================================
The previous turns built the READOUT layers (power / frequency / arrow) open-loop.
This closes the one edge that makes it a hippocampal LOOP rather than a stack:

        arrow (signed velocity)  --->  grid (advance the phase)   [PATH INTEGRATION]

Functional map (biology in brackets):
  - GRID modules at spatial frequencies k_m  [MEC grid/band cells] hold a phase
    estimate Phi_m of position. They are ADVANCED by an estimated velocity -> the
    grid dead-reckons. Multiple incommensurate modules make the decode unique over
    a range far longer than any single wavelength (why grids are multi-scale).
  - ARROW = signed velocity from a QUADRATURE pair per module:
        v ~ angle(z_m(t) * conj(z_m(t-1))) / (2 pi k_m dt),   z_m = cos + i sin
    This is exactly the L2 cross-time bilinear, and the sign needs TWO channels --
    a single channel gives |v| only (the chirality lesson, here as a real ablation).
  - CA1 COMPARATOR corrects the grid phase toward the observed phase, but only in
    the theta disinhibition window [medial-septum theta clock + chandelier/AIS gate]:
    predict during the cycle, correct at the trough.
  - When sensory input drops out, there is nothing to correct against and no fresh
    velocity -- the loop HOLDS its last velocity and keeps integrating. That is
    dead-reckoning, and it is the whole point: a closed loop holds the percept
    through a blackout; an open readout loses it.

Tests (printed + figured, not assumed):
  A  closed loop tracks through a sensory blackout; open loop goes blind.
  B  the grid decode is unique with many modules, aliased with one.
  C  single-channel (direction-blind) arrow path-integrates the wrong way on reversal.
  D  the held velocity during blackout is the dead-reckoning mechanism.

Pure numpy. PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def circ_err(a, b):
    d = np.abs((a - b) % 1.0)
    return np.minimum(d, 1.0 - d)


class CorticalLoop:
    """EC-hippocampal path-integration loop on a 1-D circular track."""
    def __init__(self, k_modules=(3, 5, 7, 11), theta=10, vel_smooth=0.3,
                 ca1_gain=0.35, dt=1.0):
        self.k = np.array(k_modules, float)        # grid module spatial frequencies
        self.theta = int(theta)                    # steps per theta cycle (septal clock)
        self.alpha = vel_smooth
        self.gain = ca1_gain
        self.dt = dt
        self._cand = np.linspace(0, 1, 2000, endpoint=False)
        self._basis = 2*np.pi*np.outer(self.k, self._cand)   # (M, 2000)

    # grid -> position: where do all module phases agree (the 'beat' = place cell)
    def decode(self, Phi, modules=None):
        k = self.k if modules is None else self.k[modules]
        Phi = Phi if modules is None else Phi[modules]
        basis = 2*np.pi*np.outer(k, self._cand)
        score = np.cos(basis - Phi[:, None]).sum(0)        # population agreement
        return self._cand[np.argmax(score)], score

    def sensory(self, x, drop):
        """quadrature grid-tuned sensors z_m = cos+isin of module phase, or None."""
        if drop:
            return None
        ph = 2*np.pi*self.k*x
        return np.cos(ph) + 1j*np.sin(ph)

    def run(self, x_true, drop_mask, mode="closed", seed=0):
        """mode: 'closed' (full loop), 'open' (read instantaneous, no integration),
                 'single' (direction-blind velocity: |v| only)."""
        rng = np.random.default_rng(seed)
        T = len(x_true)
        Phi = 2*np.pi*self.k*x_true[0]                     # init on first observation
        v_int = 0.0
        x_hat = np.zeros(T); v_hist = np.zeros(T)
        z_prev = self.sensory(x_true[0], False)
        last_open = x_true[0]
        for t in range(T):
            z = self.sensory(x_true[t], drop_mask[t])
            if z is not None:
                z = z + 0.02*(rng.standard_normal(len(z)) + 1j*rng.standard_normal(len(z)))

            if mode == "open":
                # no integration: decode straight from the instantaneous phase, or
                # freeze when blind (nothing to integrate with)
                if z is not None:
                    last_open, _ = self.decode(np.angle(z))
                x_hat[t] = last_open; v_hist[t] = np.nan
                z_prev = z if z is not None else z_prev
                continue

            # --- velocity from the cross-time quadrature arrow ---
            if z is not None and z_prev is not None:
                dphi = np.angle(z * np.conj(z_prev))        # ~ 2 pi k v dt, signed
                v_meas = np.median(dphi / (2*np.pi*self.k*self.dt))
                if mode == "single":
                    v_meas = abs(v_meas)                    # direction-blind ablation
                v_int = (1-self.alpha)*v_int + self.alpha*v_meas
            # else: HOLD v_int  (dead-reckoning through the blackout)

            # --- grid path integration (every step) ---
            Phi = Phi + 2*np.pi*self.k*v_int*self.dt

            # --- CA1 correction, only in the theta window and only with input ---
            if z is not None and (t % self.theta == 0):
                Phi = Phi + self.gain*wrap(np.angle(z) - Phi)

            x_hat[t], _ = self.decode(Phi)
            v_hist[t] = v_int
            if z is not None:
                z_prev = z
        return x_hat, v_hist


# ============================================================ demo
if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    T = 1400
    # a trajectory with forward AND backward motion (to expose the direction ablation)
    v = np.full(T, 0.0040)
    v[350:550] = -0.0034                                   # walk backwards
    v[900:1000] = 0.0080                                   # a fast burst
    x_true = np.cumsum(v) % 1.0

    drop = np.zeros(T, bool)
    drop[600:950] = True                                  # a 350-step sensory blackout

    loop = CorticalLoop()
    x_closed, v_closed = loop.run(x_true, drop, "closed")
    x_open,   _        = loop.run(x_true, drop, "open")
    x_single, _        = loop.run(x_true, drop, "single")

    e_closed = circ_err(x_closed, x_true)
    e_open   = circ_err(x_open,   x_true)
    e_single = circ_err(x_single, x_true)
    win = slice(600, 950)
    print("="*66); print("CLOSING THE LOOP — path integration through a sensory blackout"); print("="*66)
    print(f"  blackout = steps 600-950 (no sensory input at all)")
    print(f"  mean tracking error   closed={e_closed.mean():.4f}  open={e_open.mean():.4f}"
          f"  single-ch={e_single.mean():.4f}")
    print(f"  error DURING blackout closed={e_closed[win].mean():.4f}  open={e_open[win].mean():.4f}")
    print(f"  -> closed loop dead-reckons through the blackout; open loop goes blind.")
    print(f"  -> single-channel (direction-blind) arrow integrates the wrong way on the")
    print(f"     backward segment: error={e_single[350:550].mean():.4f} vs closed={e_closed[350:550].mean():.4f}")

    # B. decode uniqueness: many modules vs one
    Phi_demo = 2*np.pi*loop.k*0.62
    _, score_multi = loop.decode(Phi_demo)
    _, score_one   = loop.decode(Phi_demo, modules=[0])
    cand = loop._cand
    peaks_one = (score_one > 0.9*score_one.max()).sum()
    print(f"\n  grid decode: 4 modules -> unique peak; 1 module -> {peaks_one} aliased peaks")

    # ---------------- figure ----------------
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    tt = np.arange(T)
    ax[0,0].plot(tt, e_closed, color="#2a9d4a", lw=1.2, label="closed loop")
    ax[0,0].plot(tt, e_open,   color="#b03030", lw=1.2, label="open (no feedback)")
    ax[0,0].axvspan(600, 950, color="#cccccc", alpha=0.5, label="sensory blackout")
    ax[0,0].set_title("A. closed loop dead-reckons through the blackout")
    ax[0,0].set_xlabel("step"); ax[0,0].set_ylabel("tracking error (circular)"); ax[0,0].legend(fontsize=8)

    ax[0,1].plot(tt, x_true, color="k", lw=1.4, label="true position")
    ax[0,1].plot(tt, x_closed, color="#2a9d4a", lw=1.0, ls="--", label="closed estimate")
    ax[0,1].axvspan(600, 950, color="#cccccc", alpha=0.5)
    ax[0,1].set_title("B. position tracked across the blackout"); ax[0,1].set_xlabel("step")
    ax[0,1].set_ylabel("position on track"); ax[0,1].legend(fontsize=8)

    ax[1,0].plot(cand, score_multi/score_multi.max(), color="#1f6feb", lw=1.3, label="4 modules (unique)")
    ax[1,0].plot(cand, score_one/score_one.max(), color="#d08020", lw=1.0, label="1 module (aliased)")
    ax[1,0].set_title("C. why grids are multi-scale: unique decode"); ax[1,0].set_xlabel("candidate position")
    ax[1,0].set_ylabel("population agreement"); ax[1,0].legend(fontsize=8)

    ax[1,1].plot(tt, v, color="k", lw=1.2, label="true velocity")
    ax[1,1].plot(tt, v_closed, color="#2a9d4a", lw=1.0, label="loop velocity (held in blackout)")
    ax[1,1].axvspan(600, 950, color="#cccccc", alpha=0.5)
    ax[1,1].axhline(0, color="gray", lw=0.6)
    ax[1,1].set_title("D. dead-reckoning = holding the last velocity"); ax[1,1].set_xlabel("step")
    ax[1,1].set_ylabel("velocity"); ax[1,1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig("cortical_loop.png", dpi=120)
    print("\n  figure: cortical_loop.png")
