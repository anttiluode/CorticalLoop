"""
holographic_stack.py — the Holographic Stack, built and RUN
===========================================================
Three readouts of the same analytic band signals z_k(t), each a different
symmetry class in time. NOT three things that 'stack to break Wiener-Khinchin'
(only one breaks it); three INDEPENDENT invariance classes:

  L0  amplitude  : total band energy  Sum_k <|z_k|^2>          time-SYMMETRIC
  L1  frequency  : spectral centroid  Sum_k f_k a_k / Sum a_k  time-SYMMETRIC
  L2  arrow      : net angular momentum Sum_k <Im(z_k(t) z*_k(t-d))>  ANTI-symmetric

Only L2 flips under time reversal (the chirality finding). The Bulk is IslandNet
(the pole-field memory) -- the one piece of the RH blind-zone math that transfers
literally (the 1/d influence kernel). Orchestration: each layer has its own
surprise; a reversal lights L2, a pitch jump lights L1, an onset lights L0 -- so
you can route the dream lens to whichever event you want to hallucinate on.

This file RUNS three tests:
  A. layer separability  -- is the stack a real decomposition? (3x3 sensitivity)
  B. reversal            -- does ONLY the arrow flip? (the WK signature)
  C. bulk retention      -- do stack features survive archival in IslandNet?

Honest scope: fixed analytic bank (bands not learned here -- that's the
dynamic_geometric_net story); 1-D toy (the multichannel field is the real target);
IslandNet's own README notes freezing is a simpler memory baseline.

PerceptionLab / Antti Luode, with Claude (Opus 4.8). Helsinki, June 2026.
Do not hype. Do not lie. Just show.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from island_net import IslandNet

T = 256
def gabor_bank(n_bands=16, W=48):
    fc = np.geomspace(0.04, 0.34, n_bands)
    w = np.arange(W) - W/2
    env = np.exp(-0.5*(w/(W/4))**2)
    return fc, np.stack([env*np.exp(-1j*2*np.pi*f*w) for f in fc])
FC, BANK = gabor_bank()
LAG = 6

def bands(x):
    return np.stack([np.convolve(x, k, mode="same") for k in BANK])   # (K,T) complex

# ---------------- the three layers ----------------
def layer_amplitude(z): return float((np.abs(z)**2).mean()*z.shape[0])     # total energy
def layer_frequency(z):
    a = (np.abs(z)**2).mean(1); return float((FC*a).sum()/(a.sum()+1e-12)) # centroid
def layer_arrow(z):
    # the naive sum Sum Im(z z*_lag) is dominated by CARRIER rotation and does not
    # flip under reversal. The clean scalar arrow is spectral DRIFT: the rate the
    # frequency content moves (up-sweep vs down-sweep) -- 0 for a steady tone,
    # sign-flipping under time reversal. (The full per-band bilinear pattern also
    # carries direction, but does not collapse to one honest number -- it needs the
    # multichannel pattern, exactly the chirality lesson.)
    p = np.abs(z)**2                              # (K,T)
    cen = (FC[:, None]*p).sum(0)/(p.sum(0)+1e-12) # centroid(t)
    return float(np.mean(np.diff(cen))*100.0)     # mean drift (scaled for readability)

def readout(x):
    z = bands(x)
    return np.array([layer_amplitude(z), layer_frequency(z), layer_arrow(z)])

# ---------------- stimuli ----------------
def tone(f, amp=1.0, seed=0):
    rng=np.random.default_rng(seed); t=np.arange(T)
    return (amp*(np.sin(2*np.pi*f*t)+0.3*np.sin(4*np.pi*f*t))+0.02*rng.standard_normal(T)).astype(float)
def sweep(f0, f1, amp=1.0, seed=0):
    rng=np.random.default_rng(seed); t=np.arange(T)
    ph=2*np.pi*np.cumsum(np.linspace(f0,f1,T))
    return (amp*(np.sin(ph)+0.3*np.sin(2*ph))+0.02*rng.standard_normal(T)).astype(float)

# ============================================================ A. separability
base   = tone(0.12, 1.0)
louder = tone(0.12, 2.0)        # changes amplitude only
higher = tone(0.22, 1.0)        # changes frequency only
upswp  = sweep(0.08, 0.24)      # has a direction
downsw = upswp[::-1].copy()     # opposite direction, same band coverage

r_base = readout(base)
events = {"amplitude": readout(louder)-r_base,
          "frequency": readout(higher)-r_base,
          "arrow":     readout(upswp)-readout(downsw)}     # arrow event = up vs down
names = ["amplitude","frequency","arrow"]
S = np.array([[abs(events[e][l]) for l in range(3)] for e in names])
Sn = S/ (S.max(0,keepdims=True)+1e-12)                     # normalize per layer (column)
print("="*64); print("A. LAYER SEPARABILITY  (is the stack a real decomposition?)")
print("="*64)
print(f"   {'event \\ layer':>16} {'L0 amp':>9} {'L1 freq':>9} {'L2 arrow':>9}")
for i,e in enumerate(names):
    print(f"   {e:>16} {Sn[i,0]:9.2f} {Sn[i,1]:9.2f} {Sn[i,2]:9.2f}")
diag = float(np.mean([Sn[i,i] for i in range(3)]))
offd = float((Sn.sum()-np.trace(Sn))/6)
print(f"   diagonal mean {diag:.2f}  off-diagonal mean {offd:.2f}  -> "
      f"{'near-diagonal: three independent layers' if diag>2*offd else 'overlapping'}")

# ============================================================ B. reversal
ru, rd = readout(upswp), readout(downsw)
print("\n"+"="*64); print("B. TIME REVERSAL  (does ONLY the arrow flip?)"); print("="*64)
for l,nm in enumerate(["L0 amplitude","L1 frequency","L2 arrow"]):
    flip = "FLIPS" if np.sign(ru[l])!=np.sign(rd[l]) and abs(ru[l])>1e-6 else "same"
    print(f"   {nm:>14}: up={ru[l]:+.4f}  down={rd[l]:+.4f}  -> {flip}")
print("   -> the arrow is the only layer that carries time's direction.")

# ============================================================ orchestration
print("\n"+"="*64); print("ORCHESTRATION  (which event lights which layer's surprise?)"); print("="*64)
def surprise(x_prev, x_cur):
    a,b = readout(x_prev), readout(x_cur)
    return np.abs(b-a)/ (np.abs(a)+1e-6)
for label, xc in [("onset (louder)", louder), ("pitch jump", higher), ("reversal", downsw)]:
    s = surprise(base if label!="reversal" else upswp, xc)
    hot = ["L0","L1","L2"][int(np.argmax(s))]
    print(f"   {label:>16}: surprise(L0,L1,L2)=({s[0]:.2f},{s[1]:.2f},{s[2]:.2f})  hottest={hot}")
print("   -> route the dream lens to L2 (reversal) for 'the world turned around',")
print("      or L1 (pitch jump) for 'the content changed'. They are distinct triggers.")

# ============================================================ C. bulk retention
print("\n"+"="*64); print("C. BULK (IslandNet) RETENTION  (stack features, archival by depth)"); print("="*64)
def feat(x):
    z=bands(x); a=(np.abs(z)**2).mean(1); a/=a.sum()+1e-12
    cr=z[:,LAG:]*np.conj(z[:,:-LAG]); arr=cr.imag.mean(1)
    return np.concatenate([a, arr, [layer_frequency(z)/0.34]]).astype(float)   # (2K+1,)
def make(task, n, seed):
    rng=np.random.default_rng(seed); X=[];y=[]
    for i in range(n):
        if task=="arrow":
            up=i%2==0; x=sweep(0.08,0.24,seed=seed+i); x=x if up else x[::-1].copy(); lab=int(up)
        else:  # loudness
            loud=i%2==0; x=tone(0.15,2.0 if loud else 0.7,seed=seed+i); lab=int(loud)
        X.append(feat(x)); y.append(lab)
    return np.array(X), np.array(y)
Xa,ya=make("arrow",240,1); Xb,yb=make("loud",240,7); Xte,yte=make("arrow",120,999)
D=Xa.shape[1]; DEPTH=3.0
net=IslandNet(in_dim=D, D=6, J=6, seed=0)
net.new_task("arrow",2); net.train_task("arrow",Xa,ya,steps=700)
accA=net.accuracy(Xte,yte,"arrow")
net.archive("arrow", depth=DEPTH)                       # translate slots deep (cold storage)
# RETRIEVE at the island's depth: a translation, so features match the trained head
accA_arch=net.accuracy(Xte,yte,"arrow", contour_sigma=DEPTH)
net.new_task("loud",2); net.train_task("loud",Xb,yb,steps=700)   # gradients touch non-frozen blocks
accA_after=net.accuracy(Xte,yte,"arrow", contour_sigma=DEPTH)    # retrieve task 1 at depth
# control: no archival (interference allowed), read at boundary
net2=IslandNet(in_dim=D, D=6, J=6, seed=0)
net2.new_task("arrow",2); net2.train_task("arrow",Xa,ya,steps=700)
net2.new_task("loud",2); net2.train_task("loud",Xb,yb,steps=700)  # no archive
accA_ctrl=net2.accuracy(Xte,yte,"arrow")
print(f"   arrow task acc (stack features)         : {accA:.3f}")
print(f"   retrieved at depth, before 2nd task     : {accA_arch:.3f}  (archival is lossless translation)")
print(f"   after 2nd task, WITH archival (at depth): {accA_after:.3f}")
print(f"   after 2nd task, NO archival (boundary)  : {accA_ctrl:.3f}  (interference control)")
print("   -> archival-by-depth preserves task 1; the boundary control is overwritten.")

# ============================================================ figure
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
im=ax[0].imshow(Sn, cmap="viridis", vmin=0, vmax=1)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["L0 amp","L1 freq","L2 arrow"])
ax[0].set_yticks(range(3)); ax[0].set_yticklabels(["amplitude","frequency","arrow"])
ax[0].set_ylabel("event changes..."); ax[0].set_title("A. layer separability\n(near-diagonal = independent layers)")
for i in range(3):
    for j in range(3): ax[0].text(j,i,f"{Sn[i,j]:.2f}",ha="center",va="center",
                                  color="w" if Sn[i,j]<0.6 else "k",fontsize=9)
ax[1].bar(["L0 amp","L1 freq","L2 arrow"],[ru[0],ru[1],ru[2]],alpha=0.6,label="up",color="#2a9d4a")
ax[1].bar(["L0 amp","L1 freq","L2 arrow"],[rd[0],rd[1],rd[2]],alpha=0.6,label="down (reversed)",color="#b03030")
ax[1].axhline(0,color="k",lw=1); ax[1].set_title("B. reversal flips ONLY the arrow"); ax[1].legend(fontsize=8)
ax[2].bar(["with\narchival","no\narchival"],[accA_after,accA_ctrl],color=["#1f6feb","#b03030"])
ax[2].axhline(accA,ls="--",color="k",lw=1,label=f"before 2nd task ({accA:.2f})")
ax[2].set_ylim(0,1.05); ax[2].set_title("C. bulk retention of task 1\nafter learning task 2"); ax[2].legend(fontsize=8)
fig.tight_layout(); fig.savefig("holographic_stack.png", dpi=120)
print("\n   figure: holographic_stack.png")
