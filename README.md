# Zêta de Confinement
# 🔷 Vers la Zêta-Cohérence : Fondements et Visualisations

**Auteur : Adil HAJADI**  
**Licence : MIT**

Ce projet explore une nouvelle classe de fonctions zêta modifiées à partir de poids arithmétiques liés à l’ordre multiplicatif et aux structures modulaires, dans le but de révéler des analogies spectrales profondes entre théorie des nombres et physique des particules (modèles de confinement, résonances hadroniques, etc.).

## 📄 L'article

Le papier **"Vers la Zêta-Cohérence : Fondements et Visualisations"** propose :

- Une **fonction zêta de cohérence** \( \zeta_{\text{coh}}(s) \), pondérée par la périodicité décimale.
- Une **fonction zêta de confinement** \( \zeta_{\text{conf}}(s) \), intégrant des caractères de SU(3), un lissage gaussien et une structure d’ordre multiplicatif.
- Des comparaisons avec les spectres hadroniques réels et les signaux oscillants associés aux nombres premiers.
- Une **validation arithmétique** via la formule explicite de Riemann–von Mangoldt.
- Une **analyse fréquentielle** via la méthode de Welch.

## ⚙️ Installation & Exécution

### 1. Cloner le dépôt

```bash
git clone https://github.com/tonpseudo/Zeta-Coherence-Unification.git
cd Zeta-Coherence-Unification

pip install -r requirements.txt

python zetaduconfinement.py

| Fichier Python                      | Description                                                                                       |
| ----------------------------------- | ------------------------------------------------------------------------------------------------- |
| `ArtinZeta.py`                      | Construction de la zêta d’Artin (pondération par caractères de Dirichlet et structure modulaire). |
| `StressTest.py`                     | Tests de robustesse et de sensibilité du modèle numérique.                                        |
| `StressTestArtin.py`                | Variante du stress test appliqué à la zêta d’Artin.                                               |
| `turing_dirichlet_zeros.py`         | Localisation des zéros via la méthode de Turing pour fonctions L et variantes.                    |
| `06_weil_kernel_gaussian_arb_v5.py` | Noyau de Weil avec lissage gaussien arbitraire, utilisé pour la cohérence spectrale.              |
| `zetaduconfinement.py`              | Script principal pour générer les visualisations du papier.                                       |

🖼️ Figures et Visualisations

Les figures générées incluent :

FFT et cohérence fréquentielle
Corrélations croisées pondérées
Alignement spectral avec les spectres hadroniques (type Welch)...

📘 Licence

Ce projet est distribué sous la licence MIT. Vous êtes libre de l’utiliser, modifier et redistribuer avec attribution.

🙏 Remerciements

Merci aux travaux classiques de Riemann, Odlyzko, et aux inspirations venues de la physique mathématique contemporaine (Dyson, Weil, etc.).

Pour toute remarque, discussion ou collaboration, n’hésitez pas à me contacter via GitHub ou en commentaire de l’article.
