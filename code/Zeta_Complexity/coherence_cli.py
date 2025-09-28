# coherence_cli.py
import argparse, math, os
import numpy as np
import matplotlib.pyplot as plt

# ========= I/O =========
def load_column(path, col=None):
    """Charge un CSV/TSV et renvoie un np.array(float)."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=None, engine="python")
        if col is None:
            if "value" in df.columns:
                col = "value"
            elif len(df.columns) >= 2:
                col = df.columns[1]
            else:
                col = df.columns[0]
        return df[col].astype(float).to_numpy()
    except Exception:
        data = np.genfromtxt(path, delimiter=None, names=True, dtype=None, encoding=None)
        names = data.dtype.names
        if names is None:
            arr = np.genfromtxt(path, delimiter=None)
            if arr.ndim == 1: return arr.astype(float)
            if col is None: return arr[:, 1].astype(float) if arr.shape[1] > 1 else arr[:, 0].astype(float)
            if isinstance(col, int): return arr[:, col].astype(float)
            raise ValueError("Impossible d'inférer la colonne sans pandas. Donne --col index (0,1,2…).")
        if col is None:
            col = "value" if "value" in names else names[min(1, len(names)-1)]
        if isinstance(col, int): col = names[col]
        return np.array([row[col] for row in data], dtype=float)

def save_coherence_csv(out_path, centers, Z):
    try:
        import pandas as pd
        pd.DataFrame({"center": centers, "Z": Z}).to_csv(out_path, index=False)
    except Exception:
        np.savetxt(out_path, np.c_[centers, Z], delimiter=",", header="center,Z", comments="")

# ========= Bases & noyaux =========
def hann_window(L):
    n = np.arange(L)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(L-1))

def primes_up_to(P):
    if P < 2: return np.array([], dtype=int)
    sieve = np.ones(P+1, dtype=bool); sieve[:2] = False
    for n in range(2, int(P**0.5)+1):
        if sieve[n]: sieve[n*n:P+1:n] = False
    return np.flatnonzero(sieve)

def nonprimes_up_to(P):
    m = np.arange(2, P+1)
    primes = primes_up_to(P)
    mask = np.ones_like(m, dtype=bool)
    mask[np.isin(m, primes)] = False
    return m[mask]

def mobius_upto(N):
    mu = np.ones(N+1, dtype=int)  # mu(1)=1
    square = np.zeros(N+1, dtype=bool)
    for p in range(2, int(N**0.5)+1):
        if not square[p]:
            mu[p:N+1:p] *= -1
            mu[p*p:N+1:p*p] = 0
            square[p*p:N+1:p] = True
    for p in range(int(N**0.5)+1, N+1):
        if all(p % q for q in range(2, int(p**0.5)+1)):
            mu[p:N+1:p] *= -1
    mu[0] = 0
    return mu

def dirichlet_char_mod_q(q, a=1):
    def chi(n):
        if math.gcd(n, q) != 1: return 0.0 + 0.0j
        return np.exp(2j*np.pi*(a*(n % q))/q)
    return chi

# ---- Base "motif" (ex: 1/137 -> motif 00729927) ----
def motif_digits(s):  # garde uniquement les chiffres
    return np.array([int(c) for c in s if c.isdigit()], dtype=int)

def motif_phases(win_len, motif_str="00729927", mode="frac10", mod=137):
    d = motif_digits(motif_str)
    if d.size == 0: return np.zeros(win_len)
    reps = (win_len + d.size - 1) // d.size
    dd = np.tile(d, reps)[:win_len]
    if mode == "frac10":
        phi = 2*np.pi * (dd / 10.0)
    elif mode == "mod137":
        cum = np.cumsum(dd) % int(mod)
        phi = 2*np.pi * (cum / float(mod))
    else:
        raise ValueError("phase inconnue (frac10 | mod137)")
    return phi  # taille win_len

def build_kernel(win_len, *, basis="primes", P=200, sigma=0.6, q=7,
                 jitter=0.0, seed=0, no_primes=False,
                 motif_str="00729927", phase_mode="frac10", mod=137):
    if win_len % 2 == 0: win_len += 1
    j = np.arange(win_len)[:, None]
    rng = np.random.default_rng(seed) if jitter > 0 else None

    if basis == "walsh":
        w = np.ones(win_len)
        block = max(1, win_len // 32)
        for i in range(win_len):
            if (i // block) % 2: w[i] = -1
        return w

    if basis == "primes":
        xs = nonprimes_up_to(P) if no_primes else primes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        logs = np.log(xs)[None, :]
        if jitter > 0: logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = (xs**(-sigma/2.0))[None, :]
        return (np.cos(j * logs) * weights).sum(axis=1)

    if basis == "dirichlet":
        xs = primes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        chi = dirichlet_char_mod_q(q)
        phases = np.angle(np.array([chi(int(n)) for n in xs], dtype=np.complex128))[None, :]
        logs = np.log(xs)[None, :]
        if jitter > 0: logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = (xs**(-sigma/2.0))[None, :]
        return (np.cos(j * logs + phases) * weights).sum(axis=1)

    if basis == "primepowers":
        xs = primes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        gp, gm = [], []
        for p in xs:
            m = 2
            while p**m <= xs[-1]**2:
                gp.append(p); gm.append(m); m += 1
        if not gp: return np.zeros(win_len)
        gp, gm = np.array(gp), np.array(gm)
        logs = (gm * np.log(gp))[None, :]
        if jitter > 0: logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = (gp**gm)**(-sigma/2.0)  # 1D, broadcast ok
        return (np.cos(j * logs) * weights[None, :]).sum(axis=1)

    if basis == "mobius":
        N = max(2, P)
        mu = mobius_upto(N)
        xs = np.arange(2, N+1)
        logs = np.log(xs)[None, :]
        if jitter > 0: logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = (xs**(-sigma/2.0)) * mu[2:N+1]  # 1D
        return (np.cos(j * logs) * weights[None, :]).sum(axis=1)

    if basis == "motif":
        xs = primes_up_to(P) if not no_primes else nonprimes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        logs = np.log(xs)[None, :]
        if jitter > 0: logs = logs + rng.normal(0, jitter, size=logs.shape)
        phi = motif_phases(win_len, motif_str=motif_str, mode=phase_mode, mod=mod)[:, None]
        weights = (xs**(-sigma/2.0))[None, :]
        return (np.cos(np.arange(win_len)[:, None]*logs + phi) * weights).sum(axis=1)

    raise ValueError("basis inconnue")

# ========= Cohérence & surrogates =========
def zeta_coherence_local(trace, *, basis="primes", P=200, sigma=0.6,
                         win_len=401, step=25, n_surrogates=0, seed=0,
                         jitter=0.0, q=7, no_primes=False,
                         motif_str="00729927", phase_mode="frac10", mod=137):
    x = np.asarray(trace, dtype=float)
    L = len(x)
    if win_len % 2 == 0: win_len += 1
    if win_len > L: raise ValueError(f"win_len ({win_len}) > longueur du trace ({L})")
    W = hann_window(win_len); W /= (np.linalg.norm(W) + 1e-12)
    centers = np.arange(win_len//2, L - win_len//2, step)

    kernel = build_kernel(win_len, basis=basis, P=P, sigma=sigma, q=q,
                          jitter=jitter, seed=seed, no_primes=no_primes,
                          motif_str=motif_str, phase_mode=phase_mode, mod=mod)
    kernel = (kernel * W); kernel /= (np.linalg.norm(kernel) + 1e-12)

    Z = np.empty(len(centers), dtype=float)
    for idx, c in enumerate(centers):
        seg = x[c-win_len//2 : c+win_len//2+1]
        seg = seg - seg.mean()
        seg /= (np.linalg.norm(seg) + 1e-12)
        Z[idx] = float(np.dot(seg, kernel))

    pval = None
    if n_surrogates > 0:
        rng = np.random.default_rng(seed)
        def phase_randomize(y):
            Y = np.fft.rfft(y)
            k = np.arange(len(Y))
            mask = (k != 0) & (k != len(Y)-1)
            phases = rng.uniform(0, 2*np.pi, size=mask.sum())
            Y[mask] = np.abs(Y[mask]) * np.exp(1j*phases)
            return np.fft.irfft(Y, n=len(y))
        Znull_max = np.empty(n_surrogates, dtype=float)
        for s in range(n_surrogates):
            xs = phase_randomize(x)
            Zs = np.empty_like(Z)
            for idx, c in enumerate(centers):
                seg = xs[c-win_len//2 : c+win_len//2+1]
                seg = seg - seg.mean()
                seg /= (np.linalg.norm(seg) + 1e-12)
                Zs[idx] = float(np.dot(seg, kernel))
            Znull_max[s] = np.max(Zs)
        pval = (np.sum(Znull_max >= np.max(Z)) + 1.0) / (n_surrogates + 1.0)

    return centers, Z, pval

# ========= CLI =========
def main():
    ap = argparse.ArgumentParser(description="Cohérence zêta locale (bases arithmétiques)")
    ap.add_argument("trace", help="CSV/TSV du trace (une métrique par ligne)")
    ap.add_argument("--col", default=None, help="nom ou index de colonne (ex: 'value' ou 1)")

    ap.add_argument("--basis", default="primes",
                    choices=["primes","dirichlet","primepowers","mobius","walsh","motif"],
                    help="base de noyaux")
    ap.add_argument("--P", type=int, default=200, help="borne sur p/n")
    ap.add_argument("--sigma", type=float, default=0.6, help="poids p^{-sigma/2}")
    ap.add_argument("--q", type=int, default=7, help="modulus q pour Dirichlet")
    ap.add_argument("--win", type=int, default=401, help="taille de fenêtre (impair)")
    ap.add_argument("--step", type=int, default=25, help="pas entre fenêtres")
    ap.add_argument("--sur", type=int, default=0, help="nb de surrogates (randomisation de phase)")
    ap.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    ap.add_argument("--jitter", type=float, default=0.0, help="écart-type du jitter sur log p")
    ap.add_argument("--no-primes", action="store_true", help="composés au lieu de premiers (bases primes/motif)")
    # options base "motif"
    ap.add_argument("--motif", default="00729927", help="motif décimal (ex: 00729927)")
    ap.add_argument("--phase", default="frac10", choices=["frac10","mod137"],
                    help="mapping des digits -> phase")
    ap.add_argument("--mod", type=int, default=137, help="modulus pour phase=mod137")
    ap.add_argument("--plot", action="store_true", help="afficher la courbe")
    args = ap.parse_args()

    col = int(args.col) if (args.col is not None and str(args.col).isdigit()) else args.col
    trace = load_column(args.trace, col=col)

    centers, Z, pval = zeta_coherence_local(
        trace,
        basis=args.basis, P=args.P, sigma=args.sigma, win_len=args.win, step=args.step,
        n_surrogates=args.sur, seed=args.seed, jitter=args.jitter, q=args.q,
        no_primes=args.no_primes,
        motif_str=args.motif, phase_mode=args.phase, mod=args.mod
    )

    zmax, zmin = float(np.max(Z)), float(np.min(Z))
    c_at_max = int(centers[int(np.argmax(Z))])
    sd = float(np.std(Z, ddof=1)) if len(Z) > 1 else float("nan")
    d = zmax / sd if sd > 0 else float("nan")
    print(f"max(Z)={zmax:.4f}  min(Z)={zmin:.4f}  center@max={c_at_max}  sd={sd:.4f}  d={d:.3f}")
    if pval is not None:
        print(f"p-val (globale, max sur fenêtres) ≈ {pval:.4g}")

    root, _ = os.path.splitext(args.trace)
    tag = f"_coh_{args.basis}_P{args.P}_s{args.sigma}"
    if args.no_primes and args.basis in ("primes","motif"): tag += "_NoPrimes"
    if args.jitter > 0: tag += f"_j{args.jitter}"
    if args.basis == "dirichlet": tag += f"_q{args.q}"
    if args.basis == "motif": tag += f"_{args.phase}_mot{args.mod}"
    out_path = root + tag + ".csv"
    save_coherence_csv(out_path, centers, Z)
    print("→ écrit:", out_path)

    if args.plot:
        plt.figure(figsize=(9,3))
        plt.plot(centers, Z)
        plt.title("Cohérence zêta locale")
        plt.xlabel("itération (centre)")
        plt.ylabel("score")
        plt.tight_layout(); plt.show()

if __name__ == "__main__":
    main()
