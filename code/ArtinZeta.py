#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pipeline numérique pour l'étude des fonctions L
- Dirichlet quadratiques (q=3,4,5,7,8,11)
- Formes modulaires GL(2) (niveaux 11, 14, 15, etc.)
Sorties: figures PNG + données JSON
Dépendances: numpy, mpmath, matplotlib
"""

import os
import json
import math
from dataclasses import dataclass
from typing import Callable, List, Dict, Tuple, Optional
import numpy as np
import mpmath as mp
import matplotlib.pyplot as plt

# ======================
# Configuration globale
# ======================

@dataclass
class Config:
    """Configuration centrale pour les calculs"""
    # Précision mpmath
    dps: int = 60
    
    # Plages de t par défaut
    dirichlet_tmin: float = 0.0
    dirichlet_tmax: float = 60.0
    dirichlet_dt: float = 0.25
    
    modular_tmin: float = 0.0
    modular_tmax: float = 60.0
    modular_dt: float = 0.4
    
    # Paramètres de troncature
    dirichlet_C: float = 7.0
    dirichlet_floor_min: int = 150
    
    modular_C: float = 6.5
    modular_floor_min: int = 500
    modular_cap: int = 1500
    
    # Tolérances pour recherche de zéros
    zero_tol: float = 1e-9
    max_zero_iter: int = 100

# ======================
# Utilitaires généraux
# ======================

class MathUtils:
    """Utilitaires mathématiques"""
    
    @staticmethod
    def primes_upto(n: int) -> List[int]:
        """Génère la liste des nombres premiers ≤ n"""
        if n < 2:
            return []
        sieve = bytearray(b"\x01") * (n + 1)
        sieve[0:2] = b"\x00\x00"
        for i in range(2, int(n**0.5) + 1):
            if sieve[i]:
                sieve[i*i:n+1:i] = b"\x00" * (((n - i*i) // i) + 1)
        return [i for i, v in enumerate(sieve) if v]
    
    @staticmethod
    def legendre_symbol(a: int, p: int) -> int:
        """Symbole de Legendre (a/p)"""
        ls = pow(a % p, (p - 1) // 2, p)
        return 1 if ls == 1 else -1
    
    @staticmethod
    def bisect_zero(f: Callable, a: float, b: float, 
                   tol: float = 1e-9, maxiter: int = 100) -> Optional[float]:
        """Recherche de zéro par dichotomie"""
        fa, fb = f(a), f(b)
        
        if fa == 0:
            return a
        if fb == 0:
            return b
        if fa * fb >= 0:
            return None
        
        lo, hi = (a, b) if fa < 0 else (b, a)
        flo, fhi = (fa, fb) if fa < 0 else (fb, fa)
        
        for _ in range(maxiter):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            
            if abs(fmid) < tol or (hi - lo) < tol:
                return mid
            
            if fmid <= 0:
                lo, flo = mid, fmid
            else:
                hi, fhi = mid, fmid
        
        return 0.5 * (lo + hi)

class PlotUtils:
    """Utilitaires pour les graphiques"""
    
    @staticmethod
    def setup_plotting():
        """Configure les paramètres par défaut des plots"""
        plt.rcParams.update({
            "figure.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9
        })
    
    @staticmethod
    def save_figure(filename: str, outdir: str = "out_ltests") -> str:
        """Sauvegarde une figure avec gestion des chemins"""
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, filename)
        plt.tight_layout()
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        return path

# ======================
# Fonctions L de Dirichlet
# ======================

class DirichletL:
    """Calculs pour les fonctions L de Dirichlet"""
    
    @staticmethod
    def quadratic_character(q: int) -> Callable[[int], int]:
        """Retourne un caractère quadratique modulo q"""
        def chi(n: int) -> int:
            n = int(n)
            if math.gcd(n, q) != 1:
                return 0
            
            if q == 3:
                return 1 if (n % 3) == 1 else -1
            elif q == 4:
                return 1 if (n % 4) == 1 else -1
            elif q == 5:
                r = pow(n, (5 - 1) // 2, 5)
                return 1 if r == 1 else -1
            elif q == 7:
                r = pow(n, (7 - 1) // 2, 7)
                return 1 if r == 1 else -1
            elif q == 8:
                # Caractère modulo 8
                if n % 2 == 0:
                    return 0
                if n % 8 in [1, 7]:
                    return 1
                else:
                    return -1
            elif q == 11:
                r = pow(n, (11 - 1) // 2, 11)
                return 1 if r == 1 else -1
            else:
                # Cas général (symbole de Legendre)
                r = pow(n, (q - 1) // 2, q)
                return 1 if r == 1 else -1
        
        return chi
    
    @staticmethod
    def parity(chi: Callable[[int], int]) -> int:
        """Retourne la parité du caractère (0 si pair, 1 si impair)"""
        return 0 if chi(-1) == 1 else 1
    
    @staticmethod
    def gauss_sum(q: int, chi: Callable[[int], int]) -> complex:
        """Somme de Gauss τ(χ)"""
        return sum(chi(n) * mp.exp(2j * mp.pi * n / q) for n in range(q))
    
    @staticmethod
    def truncation_parameter(q: int, t: float, C: float = 7.0, 
                           floor_min: int = 150) -> int:
        """Calcule le paramètre de troncature pour Dirichlet"""
        return max(floor_min, int(C * math.sqrt((abs(t) + 1) * q / (2 * math.pi))))
    
    @staticmethod
    def approximate_functional_equation(s: complex, q: int, 
                                      chi: Callable[[int], int], 
                                      N: int) -> Tuple[complex, complex]:
        """
        Équation fonctionnelle approchée pour L(s,χ)
        Retourne (L(s,χ), X(s))
        """
        a = DirichletL.parity(chi)
        tau = DirichletL.gauss_sum(q, chi)
        i_pow_a = (1j) ** a
        eps = tau / (i_pow_a * mp.sqrt(q))
        
        # Facteur X(s) de l'équation fonctionnelle
        Xs = (eps * ((q / mp.pi) ** (0.5 - s)) * 
              (mp.gamma((1 - s + a) / 2) / mp.gamma((s + a) / 2)))
        
        # Sommes partielles
        S1 = sum(chi(n) * mp.power(n, -s) for n in range(1, N + 1) if chi(n) != 0)
        S2 = sum(chi(n) * mp.power(n, -(1 - s)) for n in range(1, N + 1) if chi(n) != 0)
        
        return S1 + Xs * S2, Xs
    
    @staticmethod
    def hardy_function(t: float, q: int, chi: Callable[[int], int], 
                      N: int) -> Tuple[float, complex]:
        """
        Fonction Z de Hardy réelle pour Dirichlet
        Retourne (Z_χ(t), L(1/2+it,χ))
        """
        s = mp.mpf('0.5') + 1j * mp.mpf(t)
        L_val, Xs = DirichletL.approximate_functional_equation(s, q, chi, N)
        
        a = DirichletL.parity(chi)
        factor = ((mp.mpf(q) / mp.pi) ** ((s + a) / 2) * 
                 mp.gamma((s + a) / 2))
        
        phi = mp.arg(factor)
        Z = mp.exp(1j * phi) * L_val
        
        return float(mp.re(Z)), complex(L_val)
    
    @staticmethod
    def theoretical_zero_count(T: float, q: int) -> float:
        """Comptage théorique des zéros ≤ T"""
        if T <= 0:
            return 0
        return (T / math.pi) * math.log((q * T) / (2 * math.pi * math.e))

class DirichletAnalyzer:
    """Analyseur pour les fonctions L de Dirichlet"""
    
    def __init__(self, config: Config):
        self.config = config
        self.outdir = "out_ltests"
        PlotUtils.setup_plotting()
    
    def analyze_character(self, q: int) -> Dict:
        """Analyse complète pour un caractère donné"""
        chi = DirichletL.quadratic_character(q)
        
        # Grille en t
        ts = np.arange(self.config.dirichlet_tmin, 
                      self.config.dirichlet_tmax + 1e-12, 
                      self.config.dirichlet_dt)
        
        # Calcul de Z(t) sur grille coarse
        Z_vals, zeros = self._compute_hardy_zeros(q, chi, ts)
        
        # Profil |L(1/2+it)|
        t_plot, absL = self._compute_absL_profile(q, chi)
        
        # Génération des figures
        figures = self._generate_figures(q, chi, t_plot, absL, ts, Z_vals, zeros)
        
        # Résultats
        return {
            "family": "Dirichlet",
            "q": q,
            "parity": DirichletL.parity(chi),
            "t_range": [float(self.config.dirichlet_tmin), float(self.config.dirichlet_tmax)],
            "dt": float(self.config.dirichlet_dt),
            "zeros": zeros,
            "zero_count": len(zeros),
            "figures": figures
        }
    
    def _compute_hardy_zeros(self, q: int, chi: Callable[[int], int], 
                           ts: np.ndarray) -> Tuple[List[float], List[float]]:
        """Calcule Z(t) et détecte les zéros"""
        Z_vals = []
        for t in ts:
            N_heur = DirichletL.truncation_parameter(q, t, self.config.dirichlet_C, 
                                                   self.config.dirichlet_floor_min)
            try:
                z_val, _ = DirichletL.hardy_function(t, q, chi, N_heur)
                Z_vals.append(z_val)
            except Exception:
                Z_vals.append(np.nan)
        
        # Détection des zéros
        zeros = []
        for i in range(len(ts) - 1):
            if (np.isfinite(Z_vals[i]) and np.isfinite(Z_vals[i + 1]) and
                (Z_vals[i] * Z_vals[i + 1] < 0 or Z_vals[i] == 0 or Z_vals[i + 1] == 0)):
                
                def fH(x):
                    N_loc = DirichletL.truncation_parameter(q, x, self.config.dirichlet_C + 1, 
                                                          self.config.dirichlet_floor_min)
                    return DirichletL.hardy_function(x, q, chi, N_loc)[0]
                
                zero = MathUtils.bisect_zero(fH, float(ts[i]), float(ts[i + 1]), 
                                           self.config.zero_tol, self.config.max_zero_iter)
                if zero is not None:
                    zeros.append(zero)
        
        return Z_vals, sorted(list({round(z, 10): z for z in zeros}.values()))
    
    def _compute_absL_profile(self, q: int, chi: Callable[[int], int]) -> Tuple[np.ndarray, List[float]]:
        """Calcule le profil |L(1/2+it)|"""
        t_plot = np.linspace(self.config.dirichlet_tmin, self.config.dirichlet_tmax, 1500)
        absL = []
        
        for t in t_plot:
            try:
                N_heur = DirichletL.truncation_parameter(q, t, self.config.dirichlet_C + 1, 
                                                       self.config.dirichlet_floor_min)
                _, L_val = DirichletL.hardy_function(t, q, chi, N_heur)
                absL.append(abs(L_val))
            except Exception:
                absL.append(np.nan)
        
        return t_plot, absL
    
    def _generate_figures(self, q: int, chi: Callable[[int], int], 
                         t_plot: np.ndarray, absL: List[float],
                         ts: np.ndarray, Z_vals: List[float], zeros: List[float]) -> Dict[str, str]:
        """Génère toutes les figures pour un caractère"""
        figures = {}
        
        # Figure 1: |L(1/2+it)|
        plt.figure(figsize=(11, 4))
        plt.plot(t_plot, absL)
        plt.yscale('log')
        plt.title(rf'$|L(1/2+it,\chi)|$ (Dirichlet, $q={q}$)')
        plt.xlabel('$t$')
        plt.ylabel(r'$|L|$')
        figures["absL"] = PlotUtils.save_figure(f"absL_q{q}.png", self.outdir)
        
        # Figure 2: Fonction Z(t)
        plt.figure(figsize=(11, 3.6))
        plt.plot(ts, Z_vals, lw=1.0)
        plt.scatter(zeros, [0] * len(zeros), c='red', marker='x', s=18, label='zéros')
        plt.axhline(0, color='k', lw=0.8)
        plt.title(rf'Fonction de Hardy-like $Z(t)$ (Dirichlet, $q={q}$)')
        plt.xlabel('$t$')
        plt.ylabel(r'$Z(t)$')
        plt.legend(loc='upper right')
        figures["hardy"] = PlotUtils.save_figure(f"hardy_q{q}.png", self.outdir)
        
        # Figure 3: Comptage de zéros vs théorie
        Ts = np.linspace(max(5, self.config.dirichlet_tmin + 5), 
                        self.config.dirichlet_tmax, 100)
        N_obs = [sum(1 for z in zeros if z <= T) for T in Ts]
        N_th = [DirichletL.theoretical_zero_count(T, q) for T in Ts]
        
        plt.figure(figsize=(8.8, 4.5))
        plt.plot(Ts, N_obs, label=r'$N_{\mathrm{obs}}(T)$')
        plt.plot(Ts, N_th, '--', label='asymptotique')
        plt.xlabel('$T$')
        plt.ylabel('Comptage des zéros ≤ T')
        plt.title(f'Dirichlet q={q} : $N_{{\\mathrm{{obs}}}}(T)$ vs asymptotique')
        plt.legend()
        figures["counting"] = PlotUtils.save_figure(f"counting_q{q}.png", self.outdir)
        
        return figures

# ======================
# Formes modulaires GL(2)
# ======================

class ModularFormL:
    """Calculs pour les fonctions L de formes modulaires"""
    
    # Base de données des formes modulaires
    MODULAR_FORMS = {
        '11a1': {'N': 11, 'eps': -1},
        '14a1': {'N': 14, 'eps': 1},
        '15a1': {'N': 15, 'eps': 1},
        '17a1': {'N': 17, 'eps': 1},
        '19a1': {'N': 19, 'eps': 1},
        '37a1': {'N': 37, 'eps': 1}
    }
    
    @staticmethod
    def ap_coefficient_11a1(p: int) -> int:
        """Coefficient a_p pour la courbe elliptique 11a1"""
        if p == 2:
            # Comptage manuel pour p=2
            count = 1  # Point à l'infini
            for x in range(2):
                for y in range(2):
                    if (y*y + y - (x*x*x - x*x)) % 2 == 0:
                        count += 1
            return 2 + 1 - count
        
        s = 0
        for x in range(p):
            c = (x*x*x - x*x) % p
            disc = (1 - 4*c) % p
            if disc == 0:
                s += 1
            else:
                ls = MathUtils.legendre_symbol(disc, p)
                s += ls
        return -s
    
    @staticmethod
    def build_coefficients(Mmax: int, level: int, 
                          ap_function: Callable[[int], int]) -> np.ndarray:
        """Construit les coefficients a_n par multiplicativité"""
        primes = MathUtils.primes_upto(Mmax)
        
        # Stockage des a_p
        a_p = {}
        for p in primes:
            if p == level:
                a_p[p] = -1  # Pour les formes de niveau N, a_N = -1
            else:
                a_p[p] = ap_function(p)
        
        # Plus petit facteur premier
        spf = list(range(Mmax + 1))
        for p in primes:
            for k in range(p * p, Mmax + 1, p):
                if spf[k] == k:
                    spf[k] = p
        
        # Calcul des a_n par multiplicativité
        a = np.zeros(Mmax + 1, dtype=np.int64)
        a[1] = 1
        
        for n in range(2, Mmax + 1):
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
            for p, k in factors:
                ap = a_p[p]
                if level % p == 0:  # mauvais premier
                    loc_val = ap ** k
                else:
                    if k == 1:
                        loc_val = ap
                    else:
                        # Relation de récurrence de Hecke
                        ak_2, ak_1 = 1, ap
                        for _ in range(2, k + 1):
                            ak = ap * ak_1 - p * ak_2
                            ak_2, ak_1 = ak_1, ak
                        loc_val = ak_1
                val *= loc_val
            a[n] = val
        
        return a
    
    @staticmethod
    def truncation_parameter(N: int, t: float, C: float = 6.5, 
                           floor_min: int = 500, cap: int = 1500) -> int:
        """Paramètre de troncature pour formes modulaires"""
        return min(cap, max(floor_min, int(C * math.sqrt(N * (abs(t) + 1)))))
    
    @staticmethod
    def approximate_functional_equation(s: complex, a: np.ndarray, 
                                      N: int, eps: int) -> Tuple[complex, complex, int]:
        """Équation fonctionnelle approchée pour formes modulaires"""
        M = ModularFormL.truncation_parameter(N, mp.im(s))
        M = min(M, len(a) - 1)
        
        S1 = sum(int(a[n]) * mp.power(n, -s) for n in range(1, M + 1))
        
        # Facteur X(s)
        two_pi = mp.mpf(2) * mp.pi
        Xs = eps * (N ** (1 - s)) * (two_pi ** (2 * s - 2)) * (mp.gamma(2 - s) / mp.gamma(s))
        
        S2 = sum(int(a[n]) * mp.power(n, -(2 - s)) for n in range(1, M + 1))
        
        return S1 + Xs * S2, Xs, M
    
    @staticmethod
    def completed_function(s: complex, a: np.ndarray, N: int) -> complex:
        """Fonction complétée Λ(s,f)"""
        two_pi = mp.mpf(2) * mp.pi
        Ls, _, _ = ModularFormL.approximate_functional_equation(s, a, N, 1)  # eps temporaire
        return (N ** (s / 2)) * (two_pi ** (-s)) * mp.gamma(s) * Ls
    
    @staticmethod
    def hardy_function(t: float, a: np.ndarray, N: int, eps: int) -> complex:
        """
        Fonction Z réelle pour formes modulaires
        Pour ε = -1, Z(t) = -i Λ(1/2+it) est réel
        """
        s = mp.mpf('0.5') + 1j * mp.mpf(t)
        Lambda_val = ModularFormL.completed_function(s, a, N)
        
        if eps == -1:
            return -1j * Lambda_val  # réel
        else:
            return Lambda_val  # déjà réel pour ε = +1

class ModularAnalyzer:
    """Analyseur pour les formes modulaires"""
    
    def __init__(self, config: Config):
        self.config = config
        self.outdir = "out_ltests"
        PlotUtils.setup_plotting()
    
    def analyze_form(self, form_label: str) -> Dict:
        """Analyse complète pour une forme modulaire"""
        if form_label not in ModularFormL.MODULAR_FORMS:
            raise ValueError(f"Forme modulaire {form_label} non reconnue")
        
        form_info = ModularFormL.MODULAR_FORMS[form_label]
        N, eps = form_info['N'], form_info['eps']
        
        # Construction des coefficients
        M_needed = ModularFormL.truncation_parameter(N, self.config.modular_tmax, 
                                                    C=7.5, floor_min=800, cap=1500)
        M_needed = max(M_needed, 1400)
        
        a = ModularFormL.build_coefficients(M_needed, N, ModularFormL.ap_coefficient_11a1)
        
        # Calculs principaux
        ts = np.arange(self.config.modular_tmin, self.config.modular_tmax + 1e-12, 
                      self.config.modular_dt)
        
        absL, Z_vals, zeros = self._compute_profiles_and_zeros(a, N, eps, ts)
        
        # Génération des figures
        figures = self._generate_figures(form_label, N, ts, absL, Z_vals, zeros, a)
        
        return {
            "family": "GL2_modular",
            "form": form_label,
            "level": N,
            "epsilon": eps,
            "t_range": [float(self.config.modular_tmin), float(self.config.modular_tmax)],
            "dt": float(self.config.modular_dt),
            "zeros": zeros,
            "zero_count": len(zeros),
            "figures": figures
        }
    
    def _compute_profiles_and_zeros(self, a: np.ndarray, N: int, eps: int, 
                                  ts: np.ndarray) -> Tuple[List[float], List[float], List[float]]:
        """Calcule les profils et détecte les zéros"""
        absL, Z_vals = [], []
        
        for t in ts:
            s = mp.mpf('0.5') + 1j * mp.mpf(t)
            Ls, _, _ = ModularFormL.approximate_functional_equation(s, a, N, eps)
            Z_val = ModularFormL.hardy_function(t, a, N, eps)
            
            absL.append(float(abs(Ls)))
            Z_vals.append(float(mp.re(Z_val)))
        
        # Détection des zéros
        zeros = []
        for i in range(len(ts) - 1):
            if (np.isfinite(Z_vals[i]) and np.isfinite(Z_vals[i + 1]) and
                (Z_vals[i] * Z_vals[i + 1] < 0 or Z_vals[i] == 0 or Z_vals[i + 1] == 0)):
                
                def fZ(x):
                    Z_val = ModularFormL.hardy_function(x, a, N, eps)
                    return float(mp.re(Z_val))
                
                zero = MathUtils.bisect_zero(fZ, float(ts[i]), float(ts[i + 1]), 
                                           self.config.zero_tol, self.config.max_zero_iter)
                if zero is not None:
                    zeros.append(zero)
        
        return absL, Z_vals, sorted(list({round(z, 10): z for z in zeros}.values()))
    
    def _generate_figures(self, form_label: str, N: int, ts: np.ndarray,
                         absL: List[float], Z_vals: List[float], 
                         zeros: List[float], a: np.ndarray) -> Dict[str, str]:
        """Génère les figures pour une forme modulaire"""
        figures = {}
        
        # Figure 1: |L(1/2+it)|
        plt.figure(figsize=(10.5, 4))
        plt.plot(ts, absL)
        plt.yscale('log')
        plt.xlabel('$t$')
        plt.ylabel('$|L(1/2+it)|$')
        plt.title(f'Forme modulaire {form_label} — $|L(1/2+it)|$')
        figures["absL"] = PlotUtils.save_figure(f"modular_abs_{form_label}.png", self.outdir)
        
        # Figure 2: Fonction Z(t)
        plt.figure(figsize=(10.5, 3.6))
        plt.plot(ts, Z_vals)
        plt.scatter(zeros, [0] * len(zeros), c='red', marker='x', s=18, label='zéros')
        plt.axhline(0, color='k', lw=0.8)
        plt.xlabel('$t$')
        plt.ylabel('$Z(t)$')
        plt.title(f'Fonction $Z(t)$ — {form_label}')
        plt.legend()
        figures["hardy"] = PlotUtils.save_figure(f"modular_hardy_{form_label}.png", self.outdir)
        
        # Figure 3: Histogramme des a_p
        primes = MathUtils.primes_upto(3000)
        ap_vals = [a[p] for p in primes if p < len(a)]
        
        plt.figure(figsize=(8, 4.2))
        plt.hist(ap_vals, bins=21, alpha=0.7, edgecolor='black')
        plt.title(f'Histogramme des $a_p$ — {form_label}')
        plt.xlabel('$a_p$')
        plt.ylabel('Fréquence')
        figures["hist_ap"] = PlotUtils.save_figure(f"hist_ap_{form_label}.png", self.outdir)
        
        return figures

# ======================
# Programme principal
# ======================

def main():
    """Programme principal"""
    config = Config()
    
    print("=== Pipeline numérique pour les fonctions L ===\n")
    
    # Création du répertoire de sortie
    outdir = "out_ltests"
    os.makedirs(outdir, exist_ok=True)
    
    # Sauvegarde de la configuration
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(config.__dict__, f, indent=2)
    
    # Analyse des caractères de Dirichlet
    print("== Analyse des caractères de Dirichlet ==")
    dirichlet_analyzer = DirichletAnalyzer(config)
    
    dirichlet_results = {}
    for q in [3, 4, 5, 7, 8, 11]:
        print(f"Traitement de q = {q}...")
        try:
            result = dirichlet_analyzer.analyze_character(q)
            dirichlet_results[q] = result
            print(f"  → {result['zero_count']} zéros détectés")
        except Exception as e:
            print(f"  → Erreur: {e}")
    
    # Analyse des formes modulaires
    print("\n== Analyse des formes modulaires ==")
    modular_analyzer = ModularAnalyzer(config)
    
    modular_results = {}
    for form_label in ['11a1', '14a1', '15a1']:
        print(f"Traitement de {form_label}...")
        try:
            result = modular_analyzer.analyze_form(form_label)
            modular_results[form_label] = result
            print(f"  → {result['zero_count']} zéros détectés")
        except Exception as e:
            print(f"  → Erreur: {e}")
    
    # Sauvegarde des résultats
    all_results = {
        "config": config.__dict__,
        "dirichlet": dirichlet_results,
        "modular": modular_results,
        "timestamp": str(mp.mp.now())
    }
    
    with open(os.path.join(outdir, "results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n=== Analyse terminée ===")
    print(f"Résultats sauvegardés dans: {os.path.abspath(outdir)}")
    print(f"Caractères Dirichlet analysés: {list(dirichlet_results.keys())}")
    print(f"Formes modulaires analysées: {list(modular_results.keys())}")

if __name__ == "__main__":
    main()