# file: ltests_all.py
# -*- coding: utf-8 -*-
"""
Tests numériques pour fonctions L :
- Dirichlet quadratiques (q=3,4,5) : |L|, H(t), zéros, comptage vs théorie
- Modulaire GL(2), poids 2, niveau 11 (courbe elliptique 11a1) : |L|, ImΛ, zéros, histogramme des a_p
Sorties: PNG + JSON dans ./out_ltests
Dépendances: numpy, mpmath, matplotlib
"""

import os, json, math
from dataclasses import dataclass
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# ======================
# Paramètres globaux
# ======================
mp.mp.dps = 60  # précision mpmath (augmente si tu pousses tmax)
OUTDIR = "out_ltests"
os.makedirs(OUTDIR, exist_ok=True)

# Grilles t par défaut
DIRICHLET_TMIN, DIRICHLET_TMAX, DIRICHLET_DT = 0.0, 60.0, 0.25
MOD_TMIN, MOD_TMAX, MOD_DT = 0.0, 60.0, 0.4

# Troncatures heuristiques
def trunc_dirichlet(q, t, C=7.0, floor_min=150):
    return max(floor_min, int(C*math.sqrt((abs(t)+1)*q/(2*math.pi))))
def trunc_modular(N, t, C=6.5, floor_min=500, cap=1500):
    return min(cap, max(floor_min, int(C*math.sqrt(N*(abs(t)+1)))))

# ================
# Utilitaires
# ================
def ensure_fig_defaults():
    plt.rcParams.update({"figure.dpi": 140})

def savefig(filename):
    path = os.path.join(OUTDIR, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

def bisect_zero(f, a, b, tol=1e-9, maxiter=100):
    fa, fb = f(a), f(b)
    if not (fa == 0 or fb == 0 or fa*fb < 0):
        return None
    lo, hi = a, b
    flo, fhi = fa, fb
    for _ in range(maxiter):
        mid = 0.5*(lo+hi)
        fmid = f(mid)
        if abs(fmid) < tol or (hi-lo) < tol:
            return mid
        if flo == 0: return lo
        if fhi == 0: return hi
        if flo*fmid <= 0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid
    return 0.5*(lo+hi)

# =======================================
# 1) Dirichlet quadratiques (q=3,4,5)
# =======================================
def quad_char_mod(q):
    def chi(n):
        n = int(n)
        if q == 3:
            if math.gcd(n,3) != 1: return 0
            return 1 if (n % 3)==1 else -1
        if q == 4:
            if n % 2 == 0: return 0
            return 1 if (n % 4)==1 else -1
        if q == 5:
            if math.gcd(n,5) != 1: return 0
            r = pow(n, (5-1)//2, 5)  # symbole de Legendre mod 5
            return 1 if r == 1 else -1
        return 0
    return chi

def parity(chi):
    return 0 if chi(-1) == 1 else 1

def gauss_sum(q, chi):
    s = 0+0j
    for n in range(q):
        s += chi(n) * complex(mp.e**(2j*mp.pi*(n/q)))
    return s

def L_approx_AFE_dirichlet(s, q, chi, N):
    """
    AFE symétrique (Dirichlet): L(s) ≈ sum_{n≤N} χ(n) n^{-s} + X(s) sum_{n≤N} χ(n) n^{-(1-s)}
    X(s) = ε (q/π)^{1/2 - s} Γ((1-s+a)/2)/Γ((s+a)/2)
    """
    a = parity(chi)
    tau = gauss_sum(q, chi)
    i_pow_a = (1j)**a
    eps = tau / (i_pow_a * (q**0.5))
    Xs = eps * ((q/mp.pi) ** (0.5 - s)) * (mp.gamma((1 - s + a)/2) / mp.gamma((s + a)/2))
    S1 = 0+0j; S2 = 0+0j
    for n in range(1, N+1):
        val = chi(n)
        if val == 0: continue
        nmp = mp.mpf(n)
        S1 += val * nmp**(-s)
        S2 += val * nmp**(-(1 - s))
    return S1 + Xs * S2, Xs

def H_real_dirichlet(t, q, chi, N):
    """
    Version 'Hardy-like' réelle: H(t) = e^{i φ(t)} L(1/2+it) où φ vient du facteur gamma
    On renvoie (Re H, L) pour détecter les zéros de H.
    """
    s = mp.mpf('0.5') + 1j*mp.mpf(t)
    Lval, Xs = L_approx_AFE_dirichlet(s, q, chi, N)
    a = parity(chi)
    factor = (mp.mpf(q)/mp.pi) ** ((s + a)/2) * mp.gamma((s + a)/2)
    phi = mp.arg(factor)
    H = mp.e ** (1j * phi) * Lval
    return float(mp.re(H)), complex(Lval)

def run_dirichlet_block(q=3, tmin=DIRICHLET_TMIN, tmax=DIRICHLET_TMAX, dt=DIRICHLET_DT):
    ensure_fig_defaults()
    chi = quad_char_mod(q)

    # H(t) sur grille coarse, zéros
    ts = np.arange(tmin, tmax+1e-12, dt)
    Hvals, zeros = [], []
    for tt in ts:
        Nheur = trunc_dirichlet(q, tt, C=7.0, floor_min=150)
        try:
            v,_ = H_real_dirichlet(tt, q, chi, Nheur)
            Hvals.append(v)
        except Exception:
            Hvals.append(np.nan)

    def fH(x):
        Nloc = trunc_dirichlet(q, x, C=8.0, floor_min=150)
        return H_real_dirichlet(x, q, chi, Nloc)[0]

    for i in range(len(ts)-1):
        y1, y2 = Hvals[i], Hvals[i+1]
        if not (math.isfinite(y1) and math.isfinite(y2)): continue
        if y1 == 0 or y2 == 0 or y1*y2 < 0:
            z = bisect_zero(fH, float(ts[i]), float(ts[i+1]), tol=1e-9, maxiter=100)
            if z is not None: zeros.append(z)
    zeros = sorted(list({round(z,10): z for z in zeros}.values()))

    # |L| pour la figure
    t_plot = np.linspace(tmin, tmax, 1500)
    absL = []
    for tt in t_plot:
        try:
            _, L = H_real_dirichlet(tt, q, chi, trunc_dirichlet(q, tt, C=8.0, floor_min=150))
            absL.append(abs(L))
        except Exception:
            absL.append(np.nan)

    # Figures
    plt.figure(figsize=(11,4))
    plt.plot(t_plot, absL)
    plt.yscale('log')
    plt.title(rf'$|L(1/2+it,\chi)|$ (Dirichlet, $q={q}$)')
    plt.xlabel('t'); plt.ylabel(r'$|L|$')
    p_abs = savefig(f"absL_q{q}.png")

    plt.figure(figsize=(11,3.6))
    plt.plot(ts, Hvals, lw=1.0)
    plt.scatter(zeros, [0]*len(zeros), c='red', marker='x', s=18, label='zéros')
    plt.axhline(0, color='k', lw=0.8)
    plt.title(rf'Fonction de Hardy-like $H(t)$ (Dirichlet, $q={q}$)')
    plt.xlabel('t'); plt.ylabel(r'$\mathrm{Re}\,H$'); plt.legend(loc='upper right', fontsize=9)
    p_H = savefig(f"H_q{q}.png")

    # Comptage vs théorie (q= spécifique)
    def N_theory(T, q):
        return (T/math.pi) * math.log((q*T)/(2*math.pi*math.e) + 1e-12)

    Ts = np.linspace(max(5, tmin+5), tmax, 100)
    N_obs = [sum(1 for z in zeros if z <= T) for T in Ts]
    N_th  = [N_theory(T, q) for T in Ts]
    plt.figure(figsize=(8.8,4.5))
    plt.plot(Ts, N_obs, label=r'$N_{\mathrm{obs}}(T)$')
    plt.plot(Ts, N_th, '--', label='asymptotique')
    plt.xlabel('T'); plt.ylabel('Comptage des zéros ≤ T')
    plt.title(f'Dirichlet q={q} : $N_{{\\mathrm{{obs}}}}(T)$ vs asymptotique')
    p_count = savefig(f"comptage_dirichlet_vs_theorie_q{q}.png")

    # JSON
    out = {
        "family": "Dirichlet",
        "q": q,
        "t_range": [float(tmin), float(tmax)],
        "dt": float(dt),
        "zeros": [float(z) for z in zeros],
        "figures": {
            "absL": p_abs,
            "H": p_H,
            "counting": p_count
        }
    }
    with open(os.path.join(OUTDIR, f"dirichlet_q{q}_results.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

# =======================================
# 2) Modulaire GL(2) : niveau 11 (11a1)
# =======================================
def primes_upto(n):
    sieve = bytearray(b"\x01")*(n+1)
    sieve[0:2] = b"\x00\x00"
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            step = i
            start = i*i
            sieve[start:n+1:step] = b"\x00"*(((n - start)//step)+1)
    return [i for i,v in enumerate(sieve) if v]

def legendre_symbol(a, p):
    return pow(a%p, (p-1)//2, p)

def ap_for_prime_11a1(p):
    # E: y^2 + y = x^3 - x^2 (mod p): a_p = p+1 - #E(F_p); compter via discriminant d'une quadratique en y
    if p == 2:
        pts = 1
        for x in range(p):
            for y in range(p):
                if (y*y + y - (x*x*x - x*x)) % p == 0:
                    pts += 1
        return p + 1 - pts
    s = 0
    for x in range(p):
        c = (x*x*x - x*x) % p
        disc = (1 - 4*c) % p
        ls = legendre_symbol(disc, p)
        if ls == p-1: ls = -1
        s += ls
    return -s

def build_an_11a1(Mmax, N=11):
    ps = primes_upto(Mmax)
    a_p = {p: ap_for_prime_11a1(p) for p in ps}
    a_p[11] = -1  # 11a1: reduction multiplicative, a_11 = -1

    # SPF
    spf = list(range(Mmax+1))
    for p in ps:
        for k in range(p*p, Mmax+1, p):
            if spf[k] == k:
                spf[k] = p

    a = np.zeros(Mmax+1, dtype=np.int64)
    a[1] = 1
    for n in range(2, Mmax+1):
        m = n
        factors = []
        while m > 1:
            p = spf[m]
            cnt = 0
            while m % p == 0:
                m //= p
                cnt += 1
            factors.append((p, cnt))
        val = 1
        for (p,k) in factors:
            ap = a_p.get(p, ap_for_prime_11a1(p))
            if N % p == 0:
                loc_val = ap ** k
            else:
                if k == 1:
                    loc_val = ap
                else:
                    akm2 = 1
                    akm1 = ap
                    for _ in range(2, k+1):
                        ak = ap*akm1 - p*akm2
                        akm2, akm1 = akm1, ak
                    loc_val = akm1
            val *= loc_val
        a[n] = val
    return a

def modular_X_of_s(s, N=11, eps=-1):
    two_pi = mp.mpf(2)*mp.pi
    return eps * (N ** (1 - s)) * (two_pi ** (2*s - 2)) * (mp.gamma(2 - s) / mp.gamma(s))

def L_via_AFE_modular(s, a, N=11, eps=-1):
    M = trunc_modular(N, mp.im(s))
    M = min(M, len(a)-1)
    S1 = mp.mpc(0)
    for n in range(1, M+1):
        S1 += int(a[n]) * (mp.power(n, -s))
    Xs = modular_X_of_s(s, N=N, eps=eps)
    S2 = mp.mpc(0)
    for n in range(1, M+1):
        S2 += int(a[n]) * (mp.power(n, -(2 - s)))
    return S1 + Xs * S2, Xs, M

def Lambda_of_s_modular(s, a, N=11):
    two_pi = mp.mpf(2)*mp.pi
    Ls, Xs, M = L_via_AFE_modular(s, a, N=N)
    return (N ** (s/2)) * (two_pi ** (-s)) * mp.gamma(s) * Ls

def run_modular_block(N=11, tmin=MOD_TMIN, tmax=MOD_TMAX, dt=MOD_DT):
    ensure_fig_defaults()
    # Coefficients (borne basée sur tmax)
    M_needed = trunc_modular(N, tmax, C=7.5, floor_min=800, cap=1500)
    M_needed = max(M_needed, 1400)
    a = build_an_11a1(M_needed, N=N)

    ts = np.arange(tmin, tmax+1e-12, dt)
    absL, ImLam = [], []
    M_used = []
    for t in ts:
        s = mp.mpf('0.5') + 1j*mp.mpf(t)
        Lam = Lambda_of_s_modular(s, a, N=N)
        Ls, Xs, M = L_via_AFE_modular(s, a, N=N)
        absL.append(float(abs(Ls)))
        ImLam.append(float(mp.im(Lam)))
        M_used.append(M)

    # Figures: |L| et ImΛ
    plt.figure(figsize=(10.5,4))
    plt.plot(ts, absL)
    plt.yscale('log')
    plt.xlabel('t'); plt.ylabel('|L(1/2+it)|')
    plt.title(f'Modular L (level {N}) — |L(1/2+it)| up to t={tmax:g}')
    p_abs = savefig(f"modular_L_abs_level{N}_t{int(tmax)}.png")

    plt.figure(figsize=(10.5,3.6))
    plt.plot(ts, ImLam)
    plt.axhline(0, lw=0.8)
    plt.xlabel('t'); plt.ylabel('Im Λ(1/2+it)')
    plt.title(f'Im Λ(1/2+it) up to t={tmax:g} (level {N})')
    p_im = savefig(f"modular_L_imLambda_level{N}_t{int(tmax)}.png")

    # Zéros via changement de signe d'ImΛ
    def fIm(t):
        s = mp.mpf('0.5') + 1j*mp.mpf(t)
        return float(mp.im(Lambda_of_s_modular(s, a, N=N)))
    zeros = []
    for i in range(len(ts)-1):
        y1, y2 = ImLam[i], ImLam[i+1]
        if not (math.isfinite(y1) and math.isfinite(y2)): continue
        if y1 == 0 or y2 == 0 or y1*y2 < 0:
            z = bisect_zero(fIm, float(ts[i]), float(ts[i+1]), tol=1e-9, maxiter=100)
            if z is not None: zeros.append(z)
    zeros = sorted(list({round(z,10): z for z in zeros}.values()))

    # Histogramme ap (primes ≤ 3000)
    ps = [p for p in primes_upto(3000) if p >= 3]
    ap_vals = [int(build_an_11a1(3000, N=N)[p]) for p in ps]  # simple (pas optimal), mais clair
    plt.figure(figsize=(8,4.2))
    plt.hist(ap_vals, bins=21)
    plt.title(rf'Histogramme des $a_p$ (niveau {N}, primes ≤ 3000)')
    plt.xlabel(r'$a_p$'); plt.ylabel('fréquence')
    p_hist = savefig("hist_ap_level11.png")

    # JSON
    out = {
        "family": "GL2_modular",
        "level": N,
        "precision_digits": int(mp.mp.dps),
        "t_range": [float(tmin), float(tmax)],
        "dt": float(dt),
        "M_terms_used": {"min": int(min(M_used)), "max": int(max(M_used)), "M_prebuilt": int(M_needed)},
        "zeros_critical_line": [float(z) for z in zeros],
        "absL_samples": {
            "t": [float(x) for x in ts[::5]],
            "absL": [float(absL[i]) for i in range(0, len(ts), 5)]
        },
        "figures": {
            "absL": p_abs,
            "ImLambda": p_im,
            "hist_ap": p_hist
        }
    }
    with open(os.path.join(OUTDIR, f"modular_level{N}_results_t{int(tmax)}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out

# =====================
# 3) Main (exemples)
# =====================
if __name__ == "__main__":
    print("== Dirichlet tests (q=3,4,5) ==")
    for q in (3,4,5):
        res = run_dirichlet_block(q=q)
        print(f"q={q}: {len(res['zeros'])} zeros on [0,{DIRICHLET_TMAX}] ->", res['figures'])

    print("\n== Modular GL(2) level 11 ==")
    resm = run_modular_block(N=11)
    print(f"level 11: {len(resm['zeros_critical_line'])} zeros on [0,{MOD_TMAX}] ->", resm['figures'])

    print(f"\nResults saved in: {os.path.abspath(OUTDIR)}")
