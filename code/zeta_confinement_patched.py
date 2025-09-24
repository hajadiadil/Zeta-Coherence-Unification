import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import pandas as pd

def sieve_primes(n):
    sieve = bytearray(b"\x01") * (n+1)
    sieve[:2] = b"\x00\x00"
    for p in range(2, int(n**0.5)+1):
        if sieve[p]:
            sieve[p*p:n+1:p] = b"\x00" * ((n - p*p)//p + 1)
    return [i for i in range(2, n+1) if sieve[i]]

def factorize(n):
    fac = {}
    d = 2
    while d*d <= n:
        while n % d == 0:
            fac[d] = fac.get(d, 0) + 1
            n //= d
        d += 1 if d == 2 else 2
    if n > 1:
        fac[n] = fac.get(n, 0) + 1
    return fac

def multiplicative_order_base_b_mod_p(b, p):
    if p in (2,5) or math.gcd(b, p) != 1:
        return None
    order = p - 1
    fac = factorize(p - 1)
    for q, e in fac.items():
        for _ in range(e):
            if pow(b, order // q, p) == 1:
                order //= q
            else:
                break
    return order

def chi_mod_m_k(p, m=3, k=1):
    return cmath.exp(2j*math.pi * (k * (p % m) % m) / m)

def build_ap(primes, lam=0.02, nu=0.0, b=10, m=3, k=1, use_ord_weight=True):
    logp = np.log(primes, dtype=float)
    damp = np.exp(-lam * (logp**2))
    tilt = np.exp(nu / logp)
    phase = np.array([chi_mod_m_k(int(p), m=m, k=k) for p in primes], dtype=complex)
    if use_ord_weight:
        ords = np.array([multiplicative_order_base_b_mod_p(b, int(p)) for p in primes], dtype=float)
        ords = np.where(np.isfinite(ords), ords, np.inf)
        weight = 1.0/ords
    else:
        weight = np.ones_like(damp)
    ap = damp * tilt * weight * phase
    return ap.astype(complex), logp, weight

def logabs_zeta_conf_line(sigma, tgrid, logp, ap):
    c = ap * np.exp(-sigma * logp)
    W = c[:, None] * np.exp(-1j * logp[:, None] * tgrid[None, :])
    logabs = -np.sum(np.log(np.abs(1 - W)), axis=0)
    return logabs

def main():
    Pmax = 4000
    b = 10
    lam = 0.02
    nu = 0.0
    m = 3
    k = 1
    sigma = 0.5
    tmin, tmax, Nt = 0.0, 80.0, 2500

    primes = np.array(sieve_primes(Pmax), dtype=float)
    primes = primes[(primes != 2) & (primes != 5)]

    ap, logp, ord_weight = build_ap(primes, lam=lam, nu=nu, b=b, m=m, k=k, use_ord_weight=True)

    t = np.linspace(tmin, tmax, Nt)
    logabs = logabs_zeta_conf_line(sigma, t, logp, ap)
    log10abs = logabs / math.log(10.0)

    import pandas as pd
    pd.DataFrame({"t": t, "log10|zeta_conf|": log10abs}).to_csv("profile_critical_line.csv", index=False)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,4))
    plt.plot(t, log10abs)
    plt.xlabel("t")
    plt.ylabel("log10 |ζ_conf(1/2 + i t)|")
    plt.title("Zêta de confinement — profil sur la droite critique (σ=1/2)")
    plt.tight_layout()
    plt.savefig("zeta_conf_profile.png", bbox_inches="tight")
    plt.close()

    sigma_grid = np.linspace(0.2, 0.9, 160)
    t_grid = np.linspace(0, 40, 240)
    Z = np.empty((len(sigma_grid), len(t_grid)))
    for i, sg in enumerate(sigma_grid):
        Z[i, :] = logabs_zeta_conf_line(sg, t_grid, logp, ap) / math.log(10.0)
    pd.DataFrame(Z, index=sigma_grid, columns=t_grid).to_csv("zeta_conf_heatmap.csv")

    plt.figure(figsize=(7,6))
    plt.pcolormesh(sigma_grid, t_grid, Z.T, shading='auto')
    plt.xlabel("σ = Re(s)")
    plt.ylabel("t = Im(s)")
    plt.title("log10 |ζ_conf(s)| — carte (σ,t)")
    plt.colorbar(label="log10 |ζ_conf(s)|")
    plt.tight_layout()
    plt.savefig("zeta_conf_heatmap.png", bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
