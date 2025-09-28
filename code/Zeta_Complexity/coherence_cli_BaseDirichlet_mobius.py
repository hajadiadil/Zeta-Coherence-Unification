# coherence_cli.py
import argparse, math, os
import numpy as np
import matplotlib.pyplot as plt

# ========= utils I/O =========================================================
def load_column(path, col=None):
    """
    Charge un fichier CSV/TSV et renvoie un np.array(float).
    - si pandas dispo: on l'utilise
    - sinon: numpy.genfromtxt avec auto-détection simple
    """
    try:
        import pandas as pd  # type: ignore
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
            if arr.ndim == 1:
                return arr.astype(float)
            if col is None:
                return arr[:, 1].astype(float) if arr.shape[1] > 1 else arr[:, 0].astype(float)
            if isinstance(col, int):
                return arr[:, col].astype(float)
            raise ValueError("Impossible d'inférer la colonne sans pandas. Donne --col index (0,1,2…).")
        if col is None:
            col = "value" if "value" in names else names[min(1, len(names)-1)]
        if isinstance(col, int):
            col = names[col]
        return np.array([row[col] for row in data], dtype=float)

def save_coherence_csv(out_path, centers, Z):
    try:
        import pandas as pd  # type: ignore
        pd.DataFrame({"center": centers, "Z": Z}).to_csv(out_path, index=False)
    except Exception:
        np.savetxt(out_path, np.c_[centers, Z], delimiter=",", header="center,Z", comments="")

# ========= maths: bases & noyaux ============================================
def hann_window(L):
    n = np.arange(L)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(L-1))

def primes_up_to(P):
    if P < 2: return np.array([], dtype=int)
    sieve = np.ones(P+1, dtype=bool); sieve[:2] = False
    for n in range(2, int(P**0.5)+1):
        if sieve[n]:
            sieve[n*n:P+1:n] = False
    return np.flatnonzero(sieve)

def nonprimes_up_to(P):
    m = np.arange(2, P+1)
    primes = primes_up_to(P)
    mask = np.ones_like(m, dtype=bool)
    mask[np.isin(m, primes)] = False
    return m[mask]

def mobius_upto(N):
    mu = np.ones(N+1, dtype=int)
    square = np.zeros(N+1, dtype=bool)
    for p in range(2, int(N**0.5)+1):
        if not square[p]:
            # marque les multiples de p comme changés de signe
            mu[p:N+1:p] *= -1
            # annule pour les multiples de p^2
            mu[p*p:N+1:p*p] = 0
            square[p*p:N+1:p] = True
    # traiter les premiers > sqrt(N): changent de signe
    for p in range(int(N**0.5)+1, N+1):
        if all(p % q for q in range(2, int(p**0.5)+1)):
            mu[p:N+1:p] *= -1
    mu[:2] = [0,1]  # mu(0)=0, mu(1)=1
    return mu

def dirichlet_char_mod_q(q, a=1):
    # caractère multiplicatif "jouet": χ(n)=exp(2πi a n/q) si (n,q)=1, 0 sinon
    def chi(n):
        if math.gcd(n, q) != 1:
            return 0.0 + 0.0j
        return np.exp(2j*np.pi*(a*(n % q))/q)
    return chi

def build_kernel(win_len, basis="primes", P=200, sigma=0.6, q=7, jitter=0.0, seed=None, no_primes=False):
    """
    basis: 'primes' | 'dirichlet' | 'primepowers' | 'mobius' | 'walsh'
    no_primes: baseline "composite" pour la base 'primes' (ignore si autre base)
    """
    if win_len % 2 == 0: win_len += 1
    j = np.arange(win_len)[:, None]
    rng = np.random.default_rng(seed) if (jitter>0 or seed is not None) else None

    if basis == "walsh":
        # signal binaire ±1 piècewise (contrôle non arithmétique)
        w = np.ones(win_len)
        block = max(1, win_len // 32)  # motif suffisamment rapide
        for i in range(win_len):
            if (i // block) % 2:
                w[i] = -1
        return w

    # ----- arithmétiques -----
    if basis == "primes":
        xs = nonprimes_up_to(P) if no_primes else primes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        logs = np.log(xs)[None, :]
        if jitter > 0:
            logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = xs**(-sigma/2.0)
        K = np.cos(j * logs) * weights[None, :]
        return K.sum(axis=1)

    if basis == "dirichlet":
        xs = primes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        chi = dirichlet_char_mod_q(q)
        phases = np.angle(np.array([chi(int(n)) for n in xs], dtype=np.complex128))[None, :]
        logs = np.log(xs)[None, :]
        if jitter > 0:
            logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = xs**(-sigma/2.0)
        K = np.cos(j * logs + phases) * weights[None, :]
        return K.sum(axis=1)

    if basis == "primepowers":
        xs = primes_up_to(P)
        if xs.size == 0: return np.zeros(win_len)
        grid_p, grid_m = [], []
        for p in xs:
            m = 2
            while p**m <= xs[-1]**2:  # borne simple
                grid_p.append(p); grid_m.append(m); m += 1
        if not grid_p: return np.zeros(win_len)
        grid_p, grid_m = np.array(grid_p), np.array(grid_m)
        logs = (grid_m * np.log(grid_p))[None, :]
        if jitter > 0:
            logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = (grid_p**grid_m)**(-sigma/2.0)
        K = np.cos(j * logs) * weights[None, :]
        return K.sum(axis=1)

    if basis == "mobius":
        N = max(2, P)
        mu = mobius_upto(N)
        xs = np.arange(2, N+1)
        logs = np.log(xs)[None, :]
        if jitter > 0:
            logs = logs + rng.normal(0, jitter, size=logs.shape)
        weights = (xs**(-sigma/2.0)) * mu[2:N+1]
        K = np.cos(j * logs) * weights[None, :]
        return K.sum(axis=1)

    raise ValueError("basis inconnue")

# ========= cohérence locale + surrogates =====================================
def zeta_coherence_local(trace, basis="primes", P=200, sigma=0.6, win_len=401, step=25,
                         n_surrogates=0, seed=0, jitter=0.0, q=7, no_primes=False):
    x = np.asarray(trace, dtype=float)
    L = len(x)
    if win_len % 2 == 0: win_len += 1
    if win_len > L:
        raise ValueError(f"win_len ({win_len}) > longueur du trace ({L})")
    W = hann_window(win_len); W /= (np.linalg.norm(W) + 1e-12)
    centers = np.arange(win_len//2, L - win_len//2, step)

    kernel = build_kernel(win_len, basis=basis, P=P, sigma=sigma, q=q,
                          jitter=jitter, seed=seed, no_primes=no_primes)
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

# ========= CLI ===============================================================
def main():
    ap = argparse.ArgumentParser(description="Cohérence zêta locale (bases arithmétiques)")
    ap.add_argument("trace", help="CSV/TSV du trace (une métrique par ligne)")
    ap.add_argument("--col", help="nom ou index de colonne (défaut: 'value' ou 2e colonne)")
    ap.add_argument("--basis", default="primes",
                    choices=["primes","dirichlet","primepowers","mobius","walsh"],
                    help="base de noyaux")
    ap.add_argument("--P", type=int, default=200, help="borne sur les p / n")
    ap.add_argument("--sigma", type=float, default=0.6, help="poids p^{-sigma/2}")
    ap.add_argument("--q", type=int, default=7, help="modulus pour dirichlet")
    ap.add_argument("--win", type=int, default=401, help="taille de fenêtre (impair)")
    ap.add_argument("--step", type=int, default=25, help="pas entre fenêtres")
    ap.add_argument("--sur", type=int, default=0, help="nb de surrogates (phase randomization)")
    ap.add_argument("--seed", type=int, default=0, help="graine aléatoire")
    ap.add_argument("--jitter", type=float, default=0.0, help="écart-type du jitter sur log p")
    ap.add_argument("--no-primes", action="store_true", help="baseline composés (seulement pour basis=primes)")
    ap.add_argument("--plot", action="store_true", help="afficher la courbe")
    args = ap.parse_args()

    # charger le trace
    col = int(args.col) if (args.col is not None and args.col.isdigit()) else args.col
    trace = load_column(args.trace, col=col)

    centers, Z, pval = zeta_coherence_local(
        trace,
        basis=args.basis, P=args.P, sigma=args.sigma, win_len=args.win, step=args.step,
        n_surrogates=args.sur, seed=args.seed, jitter=args.jitter, q=args.q,
        no_primes=args.no_primes
    )

    # stats rapides
    zmax, zmin = float(np.max(Z)), float(np.min(Z))
    c_at_max = int(centers[int(np.argmax(Z))])
    sd = float(np.std(Z, ddof=1)) if len(Z) > 1 else float("nan")
    effect_d = zmax / sd if sd > 0 else float("nan")

    print(f"max(Z)={zmax:.4f}  min(Z)={zmin:.4f}  center@max={c_at_max}  sd={sd:.4f}  d={effect_d:.3f}")
    if pval is not None:
        print(f"p-val (globale, max sur fenêtres) ≈ {pval:.4g}")

    # export CSV
    root, _ = os.path.splitext(args.trace)
    tag = f"_coh_{args.basis}_P{args.P}_s{args.sigma}"
    if args.no_primes and args.basis=="primes":
        tag += "_NoPrimes"
    if args.jitter>0:
        tag += f"_j{args.jitter}"
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
