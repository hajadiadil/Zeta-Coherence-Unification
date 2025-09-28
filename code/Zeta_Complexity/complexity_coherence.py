# fichier: complexity_coherence.py
import numpy as np

# --- utilitaires --------------------------------------------------------------
def primes_up_to(P):
    """Crible simple, suffisant pour P <= ~2e6."""
    if P < 2: return np.array([], dtype=int)
    sieve = np.ones(P+1, dtype=bool)
    sieve[:2] = False
    for n in range(2, int(P**0.5)+1):
        if sieve[n]:
            sieve[n*n:P+1:n] = False
    return np.flatnonzero(sieve)

def hann_window(L):
    n = np.arange(L)
    return 0.5 - 0.5*np.cos(2*np.pi*n/(L-1))

# --- noyau arithmétique + cohérence locale -----------------------------------
def zeta_coherence_local(trace, P=1000, sigma=0.5, win_len=257, step=16, seed=None,
                         n_surrogates=0):
    """
    trace: array d'entiers/réels (ex: opérations/itérations d'un algo)
    P:     borne supérieure sur les premiers utilisés
    sigma: poids p^{-sigma/2}
    win_len: taille de fenêtre (impair conseillé)
    step:  pas entre fenêtres (overlap)
    n_surrogates: si >0, calcule p-valeur par randomisation de phase
    """
    x = np.asarray(trace, dtype=float)
    L = len(x)
    if win_len % 2 == 0: win_len += 1
    W = hann_window(win_len)
    W /= np.linalg.norm(W) + 1e-12
    centers = np.arange(win_len//2, L - win_len//2, step)

    # noyau arithmétique sur la grille locale (index discret k)
    p_list = primes_up_to(P)
    weights = p_list**(-sigma/2.0)
    # matrice [win_len, nb_primes] : cos((k0+j) * log p)
    j = np.arange(win_len)[:, None]
    logp = np.log(p_list)[None, :]
    K = np.cos(j * logp)  # [win_len, nb_primes]
    Kw = K * (weights[None, :])  # pondération p^{-sigma/2}
    # noyau "combiné" (somme sur p)
    kernel = Kw.sum(axis=1)      # [win_len]
    kernel *= W
    kernel /= np.linalg.norm(kernel) + 1e-12

    # cohérence locale: corrélation normalisée entre fenêtre(x) et kernel
    Z = []
    for c in centers:
        seg = x[c-win_len//2 : c+win_len//2+1]
        seg = seg - seg.mean()
        seg /= (np.linalg.norm(seg) + 1e-12)
        Z.append(float(np.dot(seg, kernel)))
    Z = np.array(Z)

    # Surrogates par randomisation de phase (préserve PSD)
    pval = None
    if n_surrogates > 0:
        rng = np.random.default_rng(seed)
        def phase_randomize(y):
            Y = np.fft.rfft(y)
            # phases aléatoires sauf DC/nyquist
            k = np.arange(len(Y))
            mask = (k != 0) & (k != len(Y)-1)
            phases = rng.uniform(0, 2*np.pi, size=mask.sum())
            Y[mask] = np.abs(Y[mask]) * np.exp(1j*phases)
            return np.fft.irfft(Y, n=len(y))
        Z_null_max = []
        for _ in range(n_surrogates):
            xs = phase_randomize(x)
            Zs = []
            for c in centers:
                seg = xs[c-win_len//2 : c+win_len//2+1]
                seg = seg - seg.mean()
                seg /= (np.linalg.norm(seg) + 1e-12)
                Zs.append(float(np.dot(seg, kernel)))
            Z_null_max.append(np.max(Zs))
        Z_null_max = np.array(Z_null_max)
        # p-valeur (max station-wise, contrôle type "family-wise" simple)
        pval = (np.sum(Z_null_max >= np.max(Z)) + 1.0) / (n_surrogates + 1.0)

    return centers, Z, pval, kernel

# --- démo minimale ------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Exemple synthétique: trace d'un algo avec "composante arithmétique"
    n = 6000
    k = np.arange(n)
    # bruit rouge ~ 1/f : filtre passe-bas sur bruit blanc
    rng = np.random.default_rng(123)
    w = rng.normal(size=n)
    for _ in range(3):  # lissage grossier
        w = np.convolve(w, np.ones(9)/9, mode="same")
    trace = w

    # injecte une composante type cos(k log p) autour de k~4000
    p_inj = 101  # un premier
    center = 4000
    width = 500
    bump = np.exp(-0.5*((k-center)/width)**2)
    trace += 2.5 * bump * np.cos(k*np.log(p_inj))

    centers, Z, pval, kernel = zeta_coherence_local(
        trace, P=200, sigma=0.6, win_len=401, step=25, n_surrogates=200, seed=0
    )

    print(f"p-valeur (max global) ≈ {pval}")

    # tracés
    fig, ax = plt.subplots(2, 1, figsize=(10,6), sharex=False)
    ax[0].plot(trace)
    ax[0].set_title("Trace d'exécution (ex. opérations/itération)")
    ax[0].set_ylabel("Amplitude")
    ax[1].plot(centers, Z)
    ax[1].set_title("Cohérence zêta locale sur le trace (fenêtrée)")
    ax[1].set_xlabel("Itération (centre de fenêtre)")
    ax[1].set_ylabel("Score (corrélation normée)")
    plt.tight_layout()
    plt.show()
