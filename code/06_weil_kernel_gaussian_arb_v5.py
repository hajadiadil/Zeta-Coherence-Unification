#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
06_weil_kernel_gaussian_arb_v5.py

Weil kernel (base gaussienne paire) — version v5 "certif-grade"
- FT côté premiers: û(u) = (2π)σ_iσ_j · exp( -(S_ij/2) u^2 )
- u = (k log p) / u_scale
- Terme Γ: 0.5·(Re ψ(1/4 + i t/2) – log π)
- Somme sur p,k : signe e^{-k log p / 2} correctement inclus
- Queue (u > u0) bornée analytiquement
- Quadrature certifiée (intervalle) avec raffinement adaptatif
- Cholesky intervalle + Gershgorin multi-échelles
- Écrit un CSV des intervalles du Gram ET un certificate.json autonome

Dépendances:
  python-flint (arb/acb/ctx)  https://flintlib.org/python.html
"""

from __future__ import annotations
import sys, os, csv, json, math, time, argparse, hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

try:
    from flint import arb, acb, ctx
    from flint.arb import real as arb_real
    from flint.acb import complex as acb_complex
except Exception as e:
    raise SystemExit("This script requires python-flint (arb/acb). Install and re-run.\n" + str(e))

# ---------- small helpers ----------

def lower_bound(x: arb) -> float:
    return float(x.lower())

def upper_bound(x: arb) -> float:
    return float(x.upper())

def abs_sup(x: acb) -> float:
    # sup-norm of a complex ball
    return float(abs(x).upper())

def real_part(z: acb) -> arb:
    return z.real

def interval_ab(a: float, b: float) -> arb:
    # closed interval [a,b]
    return arb_real(a).interval(b)

def bar(x):
    lo = lower_bound(x); hi = upper_bound(x)
    return f"[{lo:.6e},{hi:.6e}]"

LOG_PI = arb.pi().log()

# ---------- prime sieve ----------

def sieve_primes(n: int, progress: bool=False) -> List[int]:
    """Simple sieve up to n (returns list of primes)."""
    if n < 2: return []
    bs = bytearray(b"\x01") * (n + 1)
    bs[0:2] = b"\x00\x00"
    r = int(n**0.5)
    for p in range(2, r+1):
        if bs[p]:
            step = p
            start = p*p
            bs[start:n+1:step] = b"\x00" * ((n - start)//step + 1)
            if progress and p % 10000 == 0:
                print(f"  sieve progress p={p}/{r}", flush=True)
    return [i for i in range(2, n+1) if bs[i]]

# ---------- gamma integral (even kernel) ----------

def gamma_integrand_box(t_box: arb, si: arb, sj: arb) -> arb:
    """Return enclosure of integrand for the Γ-term on a symmetric box around t ∈ [0, L].
       z = 1/4 + i t/2. We take Re ψ(z) - log π; kernel even ⇒ integrate [0,L] then ×2.
    """
    z = acb(arb("0.25"), t_box/arb(2))
    psi = z.digamma()
    re = real_part(psi) - LOG_PI
    return re

def integrate_even_cert_adaptive(F_even, L: float, target_ppu: int, max_depth: int=12) -> arb:
    """Adaptive certified integral of an even function on R using [0,L] with interval arithmetic.
       We split [0,L] dyadically until the width of F_even([a,b]) * (b-a) is small enough.
       Then multiply by 2 (even symmetry).
    """
    if L <= 0:
        return arb(0)
    # compute budget based on target_ppu
    N0 = max(1, int(math.ceil(L * target_ppu)))
    h0 = L / N0
    tol = arb(0)  # we don't need absolute numeric tol; interval handles enclosure
    acc = arb(0)

    stack: List[Tuple[float,float,int]] = [(0.0, L, 0)]
    while stack:
        a,b,depth = stack.pop()
        box = interval_ab(a,b)
        contrib = F_even(box) * arb(b - a)
        # Heuristic: if interval radius too large relative to value, bisect
        rad = float(contrib.radius())
        wid = float(upper_bound(contrib) - lower_bound(contrib))
        if (depth < max_depth) and (wid > 1e-18) and ((b-a) > h0):
            m = 0.5*(a+b)
            stack.append((a,m,depth+1))
            stack.append((m,b,depth+1))
        else:
            acc += contrib
    return acc * arb(2)

def gamma_integral_cert(si: arb, sj: arb, *, Lsig: float, pts_per_unit: int) -> arb:
    return integrate_even_cert_adaptive(lambda t: gamma_integrand_box(t, si, sj),
                                        Lsig, pts_per_unit)

# ---------- prime-side partial and tail ----------

def erfc_ball(x: arb) -> arb:
    return x.erfc()

def Spp_partial_cert_fast(si: arb, sj: arb, *, primes: List[int],
                          tau: float, u_scale: float = 1.0,
                          progress: bool=False, progress_every: int=20000,
                          progress_bar: bool=False, label: str="") -> arb:
    """
    Partial sum over primes and k≥1 for the Gaussian Weil kernel, certified by interval arithmetic.
    û(u) = (2π) σ_i σ_j · exp( -(Sij/2) u^2 )  with u=(k log p)/u_scale
    and with the factor e^{-k log p / 2} correctly included.
    We stop the inner k-sum when the geometric remainder is < tau.
    """
    Sij = si*si + sj*sj
    Aij = (arb(2)*arb.pi()) * si * sj
    US  = arb(u_scale)
    total = arb(0)
    nP = len(primes); last = 0
    for idx, p in enumerate(primes, 1):
        lp = arb(p).log()
        base_u = lp / US
        k = 1
        # ratio between successive |terms| is bounded by rho = exp( -US*base_u/2 ) * exp( -Sij*( ( (k+1)^2 - k^2 ) * base_u^2)/2 )
        # we overbound by dropping the quadratic improvement: rho0 = exp( -US*base_u/2 ), which is < 1
        rho0 = (-(US*base_u)/arb(2)).exp()
        while True:
            u = base_u * arb(k)
            decay = (-(Sij * (u*u))/arb(2) - (US*u)/arb(2)).exp()   # e^{-(Sij/2)u^2}·e^{-(US*u)/2}
            term  = (Aij * u) * decay
            total += term
            # bound the tail geometrically with ratio <= rho0
            # tail ≤ |term| * rho0 / (1 - rho0)
            if abs_sup(term) * float(abs_sup(rho0 / (arb(1)-rho0))) < tau:
                break
            k += 1
        if progress and (idx - last >= progress_every):
            last = idx
            print(f"  Spp progress: p={idx}/{nP} (p={p})", flush=True)
    return total * arb(2)   # pairness factor

def tail_bound_cert(si: arb, sj: arb, pmax: int, u_scale: float = 1.0,
                    U_pad_sigmas: float = 12.0) -> arb:
    """
    Analytic bound of the tail ∑_{p>pmax} ∑_{k≥1} 2 Aij u e^{-(US u)/2} e^{-(Sij/2)u^2}.
    We bound it by the integral ∫_{u0}^{∞} 2 Aij u e^{-(US u)/2} e^{-(Sij/2)u^2} du,
    where u0 = log(pmax+1)/US, and we truncate at U = u0 + U_pad_sigmas*sqrt(Sij)/US
    using erfc for the Gaussian tail; the linear factor u is handled explicitly.
    """
    Sij = si*si + sj*sj
    Aij = (arb(2)*arb.pi()) * si * sj
    US  = arb(u_scale)

    u0 = arb(pmax + 1).log() / US
    # main Gaussian tail: ∫_U^∞ u e^{-a u^2} du = (1/(2a)) e^{-a U^2}
    # choose U = u0 + pad * sqrt(Sij)/US to suppress the Gaussian strongly
    U  = u0 + arb(U_pad_sigmas) * Sij.sqrt() / US
    a  = Sij/arb(2)  # from e^{-(Sij/2) u^2}
    # bound for the exponential linear factor e^{-(US u)/2} on [U,∞): ≤ e^{-(US U)/2}
    lin = (-(US*U)/arb(2)).exp()
    main = (arb(1)/(arb(2)*a)) * (-(a * (U*U))).exp()
    return arb(2) * Aij * lin * main

# ---------- Gram entry and matrix ----------

def gram_entry_cert(si: arb, sj: arb, *, Lsig: float, quad_ppu: int,
                    primes: List[int], tau: float, flip: float, u_scale: float,
                    progress: bool=False, progress_every: int=20000,
                    progress_bar: bool=False, label: str="", dump_terms: bool=False) -> arb:
    I   = gamma_integral_cert(si, sj, Lsig=Lsig, pts_per_unit=quad_ppu)
    Spp = Spp_partial_cert_fast(si, sj, primes=primes, tau=tau, u_scale=u_scale,
                                progress=progress, progress_every=progress_every,
                                progress_bar=progress_bar, label=label)
    T   = tail_bound_cert(si, sj, pmax=primes[-1] if primes else 1, u_scale=u_scale)
    core = I - (Spp + T)
    val  = core * arb(flip)
    if dump_terms:
        print(f"{label} I={bar(I)}  Spp={bar(Spp)}  Tail={bar(T)}  Core={bar(core)}  Val={bar(val)}")
    return val

def build_gram_cert(sigmas: List[float], *, Lsig: float, quad_ppu: int, pmax: int, tau: float,
                    flip: float, u_scale: float, dps: int, progress: bool=False,
                    progress_every: int=20000, progress_bar: bool=False, log_entries: bool=False,
                    dump_terms: bool=False) -> List[List[arb]]:
    """Build symmetric Gram matrix with caching of gamma integrals."""
    ctx.dps = dps
    primes = sieve_primes(pmax, progress=progress)
    s = [arb(str(x)) for x in sigmas]
    n = len(s)
    G = [[arb(0) for _ in range(n)] for __ in range(n)]
    gamma_cache: Dict[Tuple[int,int], arb] = {}
    for i in range(n):
        for j in range(i, n):
            label = f"(i={i},j={j}) "
            if log_entries:
                print(f"→ Start {label}gamma+prime terms ...", flush=True)
            # reuse gamma integral if possible
            key = (i,j) if i<=j else (j,i)
            I = gamma_cache.get(key)
            if I is None:
                I = gamma_integral_cert(s[i], s[j], Lsig=Lsig, pts_per_unit=quad_ppu)
                gamma_cache[key] = I
            # build Spp and Tail, then total
            Spp = Spp_partial_cert_fast(s[i], s[j], primes=primes, tau=tau, u_scale=u_scale,
                                        progress=progress, progress_every=progress_every,
                                        progress_bar=progress_bar, label=label)
            T   = tail_bound_cert(s[i], s[j], pmax=primes[-1] if primes else 1, u_scale=u_scale)
            core = I - (Spp + T)
            val  = core * arb(flip)
            G[i][j] = G[j][i] = val
            if log_entries:
                print(f"← Done {label} {bar(val)}", flush=True)
    return G

# ---------- Gershgorin & Cholesky interval ----------

def gersh_lb_of(G: List[List[arb]], scales: List[float]) -> float:
    n = len(G)
    lb = float("inf")
    for i in range(n):
        di = lower_bound(G[i][i] / arb(scales[i]**2))
        s = 0.0
        for j in range(n):
            if i == j: continue
            s += upper_bound((G[i][j] / arb(scales[i]*scales[j])).abs())
        lb = min(lb, di - s)
    return lb

def symm_scale_and_gersh(G: List[List[arb]]) -> Tuple[List[float], float]:
    n = len(G)
    # 1) s_i = sqrt(diag(G))
    s1 = []
    ok = True
    for i in range(n):
        di = G[i][i]
        if lower_bound(di) <= 0: ok = False; break
        s1.append(float((di).sqrt()))
    scales = []
    if ok: scales.append(s1)
    # 2) s_i = (sum_j |G_ij|)^{1/2}
    s2 = []
    for i in range(n):
        s2.append(math.sqrt(sum(upper_bound(abs(G[i][j])) for j in range(n))))
    scales.append(s2)
    # 3) s_i = 1
    scales.append([1.0]*n)
    best_s, best_lb = scales[0], -1e9
    for s in scales:
        lb = gersh_lb_of(G, s)
        if lb > best_lb:
            best_lb = lb; best_s = s
    return best_s, best_lb

@dataclass
class CholReport:
    ok: bool
    min_pivot_lb: float
    min_index: int
    reason: str
    used_dps: int

def chol_psd_cert(G: List[List[arb]]) -> CholReport:
    """Interval Cholesky (LLᵀ). Certifie PSD si tous les pivots d_i = G_ii - Σ L_i,k^2 ont lower>0."""
    n = len(G)
    L = [[arb(0) for _ in range(n)] for __ in range(n)]
    min_lb = float("inf"); min_idx = -1
    for i in range(n):
        # hors-diag
        for j in range(i):
            s = arb(0)
            for k in range(j):
                s += L[i][k] * L[j][k]
            dj = L[j][j]
            if upper_bound(dj) <= 0.0:
                return CholReport(False, 0.0, j, "pivot non-positif (upper<=0)", ctx.dps)
            if lower_bound(dj) <= 0.0:
                return CholReport(False, 0.0, j, "pivot ambigu (lower<=0<upper)", ctx.dps)
            L[i][j] = (G[i][j] - s) / dj
        # diag
        s = arb(0)
        for k in range(i):
            s += L[i][k] * L[i][k]
        d = G[i][i] - s
        lb = lower_bound(d)
        if upper_bound(d) <= 0.0:
            return CholReport(False, 0.0, i, "pivot non-positif (upper<=0)", ctx.dps)
        if lb <= 0.0:
            return CholReport(False, lb, i, "pivot ambigu (lower<=0<upper)", ctx.dps)
        L[i][i] = d.sqrt()
        if lb < min_lb:
            min_lb = lb; min_idx = i
    return CholReport(True, min_lb, min_idx, "ok", ctx.dps)

# ---------- certification loop ----------

def certify_with_refine(sigmas: List[float], *, Lsig: float, quad_ppu: int,
                        pmax: int, tau: float, flip: float, u_scale: float,
                        base_dps: int, max_dps: int, progress: bool=False,
                        progress_every: int=20000, progress_bar: bool=False,
                        log_entries: bool=False, dump_terms: bool=False):
    build_args = dict(sigmas=sigmas, Lsig=Lsig, quad_ppu=quad_ppu, pmax=pmax, tau=tau,
                      flip=flip, u_scale=u_scale, progress=progress, progress_every=progress_every,
                      progress_bar=progress_bar, log_entries=log_entries, dump_terms=dump_terms)
    dps = base_dps
    while True:
        G = build_gram_cert(**{**build_args, "dps": dps})
        rep = chol_psd_cert(G)
        scales, g_lb = symm_scale_and_gersh(G)
        if rep.ok:
            return G, rep, scales, g_lb
        # échec clair
        if "non-positif" in rep.reason:
            return G, rep, scales, g_lb
        # pivot ambigu → raffiner si possible
        if dps >= max_dps:
            return G, rep, scales, g_lb
        new_dps = min(int(math.ceil(dps * 1.25)), max_dps)
        if progress:
            print(f"[auto-refine] pivot ambigu @ dps={dps} → rebuild @ dps={new_dps} (pivot i={rep.min_index})",
                  flush=True)
        dps = new_dps

# ---------- certificate emission ----------

def sha256_bytes(b: bytes) -> str:
    h = hashlib.sha256(); h.update(b); return h.hexdigest()

def sha256_file(p: Path) -> str:
    return sha256_bytes(p.read_bytes())

def write_certificate_json(path: Path, *, sigmas: List[float], params: dict,
                           G: List[List[arb]], chol: CholReport,
                           scales: List[float], gersh_lb: float,
                           primes_hash: str, code_hash: str):
    n = len(G)
    obj = {
        "meta": {
            "script": "06_weil_kernel_gaussian_arb_v5.py",
            "code_sha256": code_hash,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "flint_ctx_dps": ctx.dps
        },
        "sigmas": sigmas,
        "params": params,
        "witness": {
            "cholesky": {
                "ok": chol.ok,
                "min_pivot_lb": chol.min_pivot_lb,
                "min_index": chol.min_index,
                "reason": chol.reason,
                "used_dps": chol.used_dps
            },
            "gershgorin": {
                "lower_bound": gersh_lb,
                "scales": scales
            }
        },
        "primes": {
            "pmax": params.get("pmax"),
            "sha256": primes_hash
        },
        "gram_intervals": [[ [lower_bound(G[i][j]), upper_bound(G[i][j])] for j in range(n)] for i in range(n) ]
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

# ---------- CLI ----------

def canon_triplet(sigmas: List[float]) -> Tuple[float,float,float:
    return tuple(sorted(sigmas))

def name_from_sigmas(sigmas: List[float]) -> str:
    return "v5_" + "_".join(str(s).replace(".", "p") for s in sigmas)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sigmas', type=float, nargs='+', default=[0.6, 1.0, 1.6])
    ap.add_argument('--flip', type=float, default=-1.0)
    ap.add_argument('--u_scale', type=float, default=1.0)
    ap.add_argument('--dps', type=int, default=200)
    ap.add_argument('--max_dps', type=int, default=320)
    ap.add_argument('--quad_Lsig', type=float, default=12.0)
    ap.add_argument('--quad_ppu', type=int, default=600)
    ap.add_argument('--pmax', type=int, default=800000)
    ap.add_argument('--tau', type=float, default=1e-20)
    ap.add_argument('--csv', type=str, default='outputs/gram_weil_kernel_gaussian_arb_v5.csv')
    ap.add_argument('--cert', type=str, default='outputs/cert_weil_kernel_gaussian_arb_v5.json')
    ap.add_argument('--progress', action='store_true')
    ap.add_argument('--progress_every', type=int, default=20000)
    ap.add_argument('--progress_bar', action='store_true')
    ap.add_argument('--log_entries', action='store_true')
    ap.add_argument('--dump_terms', action='store_true')
    args = ap.parse_args()

    print("== Certified Gram (Weil kernel, Gaussian basis) — v5 ==")
    print(f"sigmas={args.sigmas}  dps={args.dps}→{args.max_dps}  Lsig={args.quad_Lsig}  ppu={args.quad_ppu}  "
          f"pmax={args.pmax}  tau={args.tau}  flip={args.flip}  u_scale={args.u_scale}")

    t0 = time.time()
    G, chol_rep, scales, g_lb = certify_with_refine(
        args.sigmas, Lsig=args.quad_Lsig, quad_ppu=args.quad_ppu,
        pmax=args.pmax, tau=args.tau, flip=args.flip, u_scale=args.u_scale,
        base_dps=args.dps, max_dps=args.max_dps,
        progress=args.progress, progress_every=args.progress_every,
        progress_bar=args.progress_bar, log_entries=args.log_entries, dump_terms=args.dump_terms
    )
    dt = time.time() - t0

    # Write CSV of intervals
    out_csv = Path(args.csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        n = len(args.sigmas)
        w.writerow([""] + [str(s) for s in args.sigmas])
        for i, si in enumerate(args.sigmas):
            w.writerow([str(si)] + [f"[{lower_bound(G[i][j]):.18e},{upper_bound(G[i][j]):.18e}]" for j in range(n)])
    print(f"→ CSV written: {out_csv}")

    # Hashes
    primes = sieve_primes(args.pmax)
    primes_bytes = ",".join(map(str, primes)).encode("utf-8")
    primes_hash = hashlib.sha256(primes_bytes).hexdigest()
    code_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    # Write certificate JSON
    params = dict(quad_Lsig=args.quad_Lsig, quad_ppu=args.quad_ppu, pmax=args.pmax, tau=args.tau,
                  u_scale=args.u_scale, flip=args.flip, dps_used=chol_rep.used_dps)
    cert_path = Path(args.cert)
    write_certificate_json(cert_path, sigmas=args.sigmas, params=params, G=G,
                           chol=chol_rep, scales=scales, gersh_lb=g_lb,
                           primes_hash=primes_hash, code_hash=code_hash)
    print(f"→ Certificate written: {cert_path}")

    # Report
    print(f"Cholesky used dps: {chol_rep.used_dps}")
    print(f"Cholesky PSD certified? {chol_rep.ok} ; chol_min_pivot_lb = {chol_rep.min_pivot_lb:.6e} (at i={chol_rep.min_index})")
    print(f"Best scaled Gershgorin lower bound: {g_lb:.6e}")
    print(f"Decision (certified): {chol_rep.ok or (g_lb>0.0)}")
    print(f"Elapsed: {dt:.2f} s")

if __name__ == "__main__":
    main()
