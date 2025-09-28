# plot_compare_137.py
import sys, os
import argparse
import numpy as np
import matplotlib.pyplot as plt

def load_center_Z(path):
    # Lecture souple: on accepte "center,Z" ou deux premières colonnes
    import csv
    centers, Z = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    # détecter header
    start = 1 if rows and any(h.lower() in ("center","z") for h in rows[0]) else 0
    for r in rows[start:]:
        if len(r) < 2:
            continue
        try:
            c = float(r[0]); z = float(r[1])
            centers.append(c); Z.append(z)
        except:
            continue
    return np.array(centers, dtype=float), np.array(Z, dtype=float)

def main():
    ap = argparse.ArgumentParser(description="Superposition Base137 vs NoPrimes137")
    ap.add_argument("base137_csv", help="CSV (center,Z) pour motif 1/137 (primes)")
    ap.add_argument("noprimes_csv", help="CSV (center,Z) pour motif 1/137 (composés)")
    ap.add_argument("--labelA", default="Base137 (primes)", help="légende série A")
    ap.add_argument("--labelB", default="NoPrimes137 (composés)", help="légende série B")
    ap.add_argument("--out", default="compare_137.png", help="nom du PNG de sortie")
    ap.add_argument("--smooth", type=int, default=0,
                    help="lissage (taille fenêtre moyenne glissante, 0=aucun)")
    args = ap.parse_args()

    cA, zA = load_center_Z(args.base137_csv)
    cB, zB = load_center_Z(args.noprimes_csv)

    if args.smooth and args.smooth > 1:
        k = args.smooth
        ker = np.ones(k)/k
        # on garde les centres d'origine; on lisse juste Z
        zA = np.convolve(zA, ker, mode="same")
        zB = np.convolve(zB, ker, mode="same")

    plt.figure(figsize=(10,4))
    plt.plot(cA, zA, label=args.labelA)
    plt.plot(cB, zB, label=args.labelB)
    plt.title("Cohérence locale — Base137 vs NoPrimes137")
    plt.xlabel("centre de fenêtre (itération)")
    plt.ylabel("score Z")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(args.out, dpi=160)
    print("→ Image écrite :", os.path.abspath(args.out))
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
