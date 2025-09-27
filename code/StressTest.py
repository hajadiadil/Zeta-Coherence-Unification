# artin_abelian_zeros.py
# Numérisation des zéros pour Artin abéliens (via Dirichlet quadratiques)
# Dépendances : mpmath, numpy, matplotlib

import os, json, math, time
import numpy as np
from mpmath import mp, mpc, mpf, gamma

# --- paramètres à ajuster ---
mp.dps = 60         # précision (10-60); augmenter si tmax grand
outdir = "artin_abelian_zeros"
q_list = [3,4,5]    # modules tests (quadratiques)
tmin, tmax = 0.0, 200.0
coarse_dt = 0.5
N_min = 150         # troncature minimale AFE (augmenter pour plus de précision)

# --- utilitaires (caractères quadratiques simples) ---
def quad_char_mod(q):
    def chi(n):
        n = int(n)
        if q == 3:
            if math.gcd(n,3) != 1: return 0
            r = n % 3
            return 1 if r==1 else -1
        if q == 4:
            if n % 2 == 0: return 0
            r = n % 4
            return 1 if r==1 else -1
        if q == 5:
            if math.gcd(n,5) != 1: return 0
            r = pow(n, (5-1)//2, 5)
            return 1 if r==1 else -1
        return 0
    return chi

def gauss_sum(q, chi):
    s = mpc(0)
    for n in range(q):
        s += chi(n) * mp.e**(2j * mp.pi * (n / q))
    return s

def parity(chi):
    return 0 if chi(-1) == 1 else 1

# AFE symétrique (retourne Lval et facteur Xs)
def L_approx_AFE(s, q, chi, N):
    a = parity(chi)
    tau = gauss_sum(q, chi)
    i_pow_a = (1j)**a
    eps = tau / (i_pow_a * mp.sqrt(q))
    Xs = eps * (mp.mpf(q)/mp.pi) ** (mp.mpf(0.5) - s) * gamma((1 - s + a)/2) / gamma((s + a)/2)
    S1 = mpc(0); S2 = mpc(0)
    N = int(N)
    for n in range(1, N+1):
        ch = chi(n)
        if ch == 0: continue
        nmp = mp.mpf(n)
        S1 += ch * nmp ** (-s)
        S2 += ch * nmp ** (-(1 - s))
    return S1 + Xs * S2, Xs

def H_real(t, q, chi, N):
    s = mp.mpf(1)/2 + mp.j * mp.mpf(t)
    Lval, Xs = L_approx_AFE(s, q, chi, N)
    a = parity(chi)
    factor = (mp.mpf(q)/mp.pi) ** ((s + a)/2) * gamma((s + a)/2)
    phi = mp.arg(factor)
    H = mp.e ** (mp.j * phi) * Lval
    return float(mp.re(H)), complex(Lval)

def bisect_zero(f, a, b, tol=1e-9, maxiter=80):
    fa = f(a); fb = f(b)
    if not (fa*fb <= 0): return None
    lo, hi = a, b; flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = 0.5*(lo+hi); fmid = f(mid)
        if abs(fmid) < tol: return mid
        if flo * fmid <= 0:
            hi = mid; fhi = fmid
        else:
            lo = mid; flo = fmid
        if hi - lo < tol: return 0.5*(lo+hi)
    return 0.5*(lo+hi)

def find_zeros(q, chi, tmin, tmax, coarse_dt):
    ts = np.arange(tmin, tmax+1e-12, coarse_dt)
    vals = []
    for tt in ts:
        Nheur = max(N_min, int(math.sqrt((abs(tt)+1)*q/(2*math.pi))*5))
        try:
            v,_ = H_real(tt, q, chi, Nheur)
            vals.append(v)
        except Exception:
            vals.append(float('nan'))
    zeros = []
    for i in range(len(ts)-1):
        v1 = vals[i]; v2 = vals[i+1]
        if not (math.isfinite(v1) and math.isfinite(v2)): continue
        if v1==0 or v2==0 or v1*v2 < 0:
            a = float(ts[i]); b = float(ts[i+1])
            f = lambda x: H_real(x, q, chi, max(N_min, int(math.sqrt((abs(x)+1)*q/(2*math.pi))*6)))[0]
            root = bisect_zero(f, a, b, tol=1e-9, maxiter=100)
            if root is not None and 0 < root <= tmax:
                zeros.append(round(root,10))
    zeros = sorted(list(set(zeros)))
    return zeros

# ---- exécution ----
if os.path.exists(outdir):
    import shutil
    shutil.rmtree(outdir)
os.makedirs(outdir)

import matplotlib.pyplot as plt
summary = {}
t_plot = np.linspace(tmin, tmax, 2001)

for q in q_list:
    print("Processing q=", q)
    chi = quad_char_mod(q)
    zeros = find_zeros(q, chi, tmin, tmax, coarse_dt)
    summary[q] = {"count": len(zeros), "zeros": zeros}
    # compute arrays for plots
    Hplot = []; absL = []
    for tt in t_plot:
        try:
            h,L = H_real(tt, q, chi, max(N_min, int(math.sqrt((abs(tt)+1)*q/(2*math.pi))*6)))
            Hplot.append(h); absL.append(abs(L))
        except Exception:
            Hplot.append(np.nan); absL.append(np.nan)
    # save pictures
    plt.figure(figsize=(12,4)); plt.plot(t_plot, absL); plt.yscale('log'); plt.title(f'|L(1/2+it)| approx q={q}'); plt.xlabel('t'); plt.ylabel('|L|'); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"absL_q{q}.png"), dpi=150); plt.close()
    plt.figure(figsize=(12,3)); plt.plot(t_plot, Hplot); plt.scatter(zeros, [0]*len(zeros), color='red', marker='x'); plt.title(f'Hardy-like Re H(t) q={q}'); plt.xlabel('t'); plt.ylabel('Re H'); plt.tight_layout(); plt.savefig(os.path.join(outdir, f"H_q{q}.png"), dpi=150); plt.close()

with open(os.path.join(outdir, "artin_abelian_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Done. Results in", outdir, "Elapsed:", time.time()-start_time)
