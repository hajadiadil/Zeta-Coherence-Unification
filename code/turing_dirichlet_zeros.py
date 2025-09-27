# turing_dirichlet_zeros.py
# Validation numérique (approximate functional equation + méthode "Turing-like")
# - évalue L(s,chi) par AFE, construit Lambda(s)
# - calcule changement d'argument d'Lambda(1/2+it) (approche Turing)
# - cherche zéros sur la ligne critique par bisection + raffinement
#
# Dépendances: mpmath, numpy, matplotlib

import os, json, math, time
import numpy as np
from mpmath import mp, mpc, mpf, gamma, log

mp.dps = 70  # précision (augmenter si nécessaire)

# --- paramètres utilisateur (modifiable) ---
outdir = "turing_dirichlet_out"
os.makedirs(outdir, exist_ok=True)

q_list = [3,5,11]         # modules à tester (exemples)
tmin, tmax = 0.0, 200.0
dt_arg = 0.5              # pas pour l'échantillonnage d'argument (plus fin = meilleure précision)
search_window = 0.8       # fenêtre autour d'un point de départ pour recherche locale de zéro
N_min = 200               # taille de troncature minimale pour AFE (augmenter pour plus de précision)

# --- utilitaires: caractères quadratiques (exemples) ---
def quad_char(q):
    def chi(n):
        n = int(n)
        if math.gcd(n, q) != 1:
            return 0
        res = pow(n % q, (q-1)//2, q)
        if res == 1: return 1
        if res == q-1: return -1
        return 0
    return chi

def gauss_sum(q, chi):
    s = mpc(0)
    for n in range(q):
        s += chi(n) * mp.e**(2j * mp.pi * (n / q))
    return s

def parity(chi):
    # a = 0 si chi(-1)=1 (pair), a=1 sinon (impair)
    return 0 if chi(-1) == 1 else 1

# --- Approximate Functional Equation (symétrique) pour L(s,chi) ---
def L_approx_AFE(s, q, chi, N):
    a = parity(chi)
    tau = gauss_sum(q, chi)
    i_pow_a = (1j)**a
    eps = tau / (i_pow_a * mp.sqrt(q))
    # facteur X(s)
    Xs = eps * (mp.mpf(q)/mp.pi) ** (mp.mpf(0.5) - s) * gamma((1 - s + a)/2) / gamma((s + a)/2)
    # somme directe et somme duale (tronquées)
    N = int(N)
    S1 = mpc(0); S2 = mpc(0)
    # la complexité vient des exposants non entiers; mpmath gère cela
    for n in range(1, N+1):
        ch = chi(n)
        if ch == 0:
            continue
        nmp = mp.mpf(n)
        S1 += ch * nmp ** (-s)
        S2 += ch * nmp ** (-(1 - s))
    return S1 + Xs * S2, Xs

# --- Lambda(s) : fonction complétée ---
def Lambda(s, q, chi, N):
    a = parity(chi)
    Lval, Xs = L_approx_AFE(s, q, chi, N)
    # facteur gamma multiplicatif (ce que PARI nomme souvent 'gamma-factor')
    Gamma_factor = (mp.mpf(q)/mp.pi) ** ((s + a)/2) * gamma((s + a)/2)
    return Gamma_factor * Lval

# --- mesure du changement d'argument de Lambda sur [t0, t1] ---
def arg_variation_Lambda(t0, t1, q, chi, N, dt=0.25):
    # échantillonne t et calcule arg(Lambda(1/2 + i t)) en unwrapping continu
    ts = np.arange(t0, t1 + 1e-12, dt)
    args = []
    for tt in ts:
        s = mp.mpf(1)/2 + mp.j * mp.mpf(tt)
        try:
            val = Lambda(s, q, chi, N)
            a = complex(val)
            args.append(math.atan2(a.imag, a.real))
        except Exception:
            args.append(0.0)
    # unwrap angle (numPy style)
    args = np.array(args)
    unwrapped = np.unwrap(args)
    delta = unwrapped[-1] - unwrapped[0]
    # changement d'argument en nombre de π
    nzeros_est = delta / math.pi
    return float(delta), float(nzeros_est), ts, args

# --- fonction "Hardy-like" H(t) variant: multiplie L par la phase gamma pour obtenir réel sur zéros ---
def H_real(t, q, chi, N):
    s = mp.mpf(1)/2 + mp.j * mp.mpf(t)
    Lval, Xs = L_approx_AFE(s, q, chi, N)
    a = parity(chi)
    factor = (mp.mpf(q)/mp.pi) ** ((s + a)/2) * gamma((s + a)/2)
    phi = mp.arg(factor)
    H = mp.e ** (mp.j * phi) * Lval
    return float(mp.re(H)), complex(Lval)

# --- recherche de zéros par détection de changement de signe + bissection ---
def bisect_zero(f, a, b, tol=1e-8, maxiter=60):
    fa = f(a); fb = f(b)
    if not (fa*fb <= 0):
        return None
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = 0.5*(lo+hi)
        fmid = f(mid)
        if abs(fmid) < tol:
            return mid
        if flo * fmid <= 0:
            hi = mid; fhi = fmid
        else:
            lo = mid; flo = fmid
        if hi - lo < tol:
            return 0.5*(lo+hi)
    return 0.5*(lo+hi)

# --- wrapper de recherche globale de zéros sur [tmin,tmax] avec pas coarse ---
def find_zeros_on_interval(q, chi, tmin, tmax, coarse_dt=0.5, Nheur_base=300):
    ts_coarse = np.arange(tmin, tmax+1e-12, coarse_dt)
    vals = []
    for tt in ts_coarse:
        Nheur = max(N_min, int(max(100, math.sqrt((abs(tt)+1)*q/(2*math.pi)) * 4)))
        try:
            v, _ = H_real(tt, q, chi, Nheur)
            vals.append(v)
        except Exception:
            vals.append(float('nan'))
    zeros = []
    for i in range(len(ts_coarse)-1):
        v1 = vals[i]; v2 = vals[i+1]
        if not (math.isfinite(v1) and math.isfinite(v2)): continue
        if v1 == 0 or v2 == 0 or v1 * v2 < 0:
            a = float(ts_coarse[i]); b = float(ts_coarse[i+1])
            f = lambda x: H_real(x, q, chi, max(N_min, int(math.sqrt((abs(x)+1)*q/(2*math.pi))*5)))[0]
            root = bisect_zero(f, a, b, tol=1e-9, maxiter=80)
            if root is not None:
                zeros.append(round(root, 10))
    # uniq + sort
    zeros = sorted(list(set(zeros)))
    return zeros

# ------------------ exécution principale ------------------
start = time.time()
summary = {}

for q in q_list:
    print("Processing q =", q)
    chi = quad_char(q)
    # 1) estimation du changement d'argument sur la fenêtre entière
    N_for_arg = max(N_min, 400)
    delta_arg, nzeros_est, ts_samples, arg_samples = arg_variation_Lambda(tmin, tmax, q, chi, N_for_arg, dt=dt_arg)
    print("  arg change (rad) =", delta_arg, " -> est. zeros =", nzeros_est)
    # 2) recherche explicite de zéros par bissection
    zeros = find_zeros_on_interval(q, chi, tmin, tmax, coarse_dt=0.5, Nheur_base=N_for_arg)
    print("  zeros found (count) =", len(zeros))
    # 3) sauvegardes graphiques : |L| et H_real with zeros marked
    # compute arrays for plotting
    t_plot = np.linspace(tmin, tmax, 2001)
    absL = []
    Hvals = []
    for tt in t_plot:
        Nheur = max(N_min, int(math.sqrt((abs(tt)+1)*q/(2*math.pi))*4))
        try:
            h, Lval = H_real(tt, q, chi, Nheur)
            Hvals.append(h)
            absL.append(abs(Lval))
        except Exception:
            Hvals.append(np.nan)
            absL.append(np.nan)
    # plot |L(1/2+it)|
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,4))
    plt.plot(t_plot, absL, label=f'|L(1/2+it, chi mod {q})| (approx)')
    plt.yscale('log')
    plt.xlabel('t'); plt.ylabel('|L| (log scale)')
    plt.title(f'|L(1/2+it)| approx, q={q}')
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"absL_q{q}.png"), dpi=150)
    plt.close()
    # plot H_real with zeros marked
    plt.figure(figsize=(12,3))
    plt.plot(t_plot, Hvals, label='Re(H(t))')
    if zeros:
        plt.scatter(zeros, [0.0]*len(zeros), marker='x', color='red', label='zeros detected')
    plt.xlabel('t'); plt.ylabel('Re H(t)')
    plt.title(f'Hardy-like Re(H(t)) (coarse) with detected zeros, q={q}')
    plt.grid(True); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"H_q{q}.png"), dpi=150)
    plt.close()
    # plot arg(Lambda) sample (coarse)
    plt.figure(figsize=(12,3))
    plt.plot(ts_samples, arg_samples, '.', markersize=3)
    plt.xlabel('t'); plt.ylabel('arg(Lambda)')
    plt.title(f'arg(Lambda(1/2+it)) samples, q={q}')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"argLambda_q{q}.png"), dpi=150)
    plt.close()

    summary[q] = {
        "arg_delta": float(delta_arg),
        "nzeros_est_from_arg": float(nzeros_est),
        "zeros_found": zeros,
        "files": [f"absL_q{q}.png", f"H_q{q}.png", f"argLambda_q{q}.png"]
    }

# save summary
with open(os.path.join(outdir, "summary_turing_dirichlet.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Done in", time.time()-start, "s. Results written in", outdir)
