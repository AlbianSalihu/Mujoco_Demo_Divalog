# 🤖 Asservissement Visuel pour la Manipulation Robotique

Système d'asservissement visuel en temps réel pour un bras robotique Panda suivant une cible mobile avec simulation MuJoCo. Démontre la cinématique inverse, le contrôle de trajectoire fluide et le retour visuel.

## 🎥 Démo
![Robot Demo](Mujoco_demo.gif)

## ✨ Fonctionnalités
- **Cinématique Inverse 6-DDL**: Solveur par moindres carrés amortis avec amortissement adaptatif
- **Boucle de Retour Visuel**: Suivi par caméra aérienne avec OpenCV
- **Contrôle de Mouvement Fluide**: Lissage exponentiel et limitation de vitesse
- **Contrôle d'Orientation Adaptatif**: Pondération dynamique basée sur la distance à la cible
- **Visualisation Temps Réel**: Flux caméra en direct avec détection de cible

## 🚀 Démarrage Rapide

### Prérequis
- Python 3.8+
- pip

### Installation
```bash
git clone https://github.com/AlbianSalihu/Mujoco_Demo_Divalog.git
cd Mujoco_Demo
pip install -r requirements.txt
```

### Exécution
```bash
python src/main.py
```

### Commandes
- **ESPACE/S**: Démarrer le suivi
- **P**: Pause
- **R**: Réinitialiser à la position d'origine
- **ESC**: Quitter

## 🏗️ Structure du Projet
```
src/
├── main.py           # Boucle de contrôle principale & machine à états
├── ik_control.py     # Solveur de cinématique inverse
├── vision.py         # Rendu caméra & détection d'objets
├── sim.py            # Configuration de la scène MuJoCo
├── config.py         # Paramètres de configuration
└── utils.py          # Fonctions d'orientation & utilitaires
```

## 🧠 Analyse Technique Approfondie

### Solveur de Cinématique Inverse
- **Algorithme**: Moindres Carrés Amortis (DLS) avec amortissement adaptatif
- **Gestion des Singularités**: λ = λ₀(1 + α‖e‖) prévient l'instabilité numérique
- **Recherche Linéaire**: Retour en arrière pour assurer la convergence
- **Point de Contrôle**: TCP calculé comme point médian entre les doigts du préhenseur
```python
# Équation IK de base (simplifiée)
dq = J^T (JJ^T + λI)^(-1) e
```

### Boucle d'Asservissement Visuel
1. **Capture**: Rendu de la vue caméra aérienne
2. **Détection**: Détection de cible basée sur OpenCV (filtrage couleur HSV)
3. **Suivi**: Lissage exponentiel: `x_t = (1-α)x_{t-1} + αx_mesuré`
4. **Résolution IK**: Calcul des vitesses articulaires pour atteindre la cible
5. **Exécution**: Application des commandes articulaires avec limitation de vitesse

### Paramètres Clés
- `step_size: 0.1` - Contrôle la vitesse de mouvement (plus bas = plus fluide/lent)
- `dq_limit: 0.01` - Vitesse articulaire maximale par itération
- `smooth_alpha: 0.1` - Facteur de lissage de cible (plus bas = plus de filtrage)
- `follow_height: 0.2` - Hauteur au-dessus de la cible (mètres)

## 📊 Métriques de Performance
- **Fréquence de Contrôle**: 100 Hz
- **Erreur de Convergence**: <5mm typique
- **Itérations IK**: 6 par cycle de contrôle
- **Latence**: ~10ms de bout en bout

## 🎛️ Guide de Réglage

### Rendre plus lent/fluide:
```python
# Dans ik_control.py -> IKConfig
step_size = 0.05      # Très lent
dq_limit = 0.005      # Très doux

# Dans main.py -> ControlConfig
smooth_alpha = 0.05   # Lissage important
```

### Rendre plus rapide/réactif:
```python
step_size = 0.3       # Rapide
dq_limit = 0.03       # Agressif
smooth_alpha = 0.3    # Moins de filtrage
```

## 🔬 Contexte Mathématique

### Moindres Carrés Amortis
Résout le système sous-déterminé `J dq = e` où:
- `J` ∈ ℝ^(6×n): Matrice jacobienne
- `dq` ∈ ℝ^n: Vitesses articulaires
- `e` ∈ ℝ^6: Erreur de position + orientation

DLS ajoute un amortissement pour gérer les singularités:
```
dq = J^T (JJ^T + λI)^(-1) e
```

### Lissage Exponentiel
Filtre les positions de cible bruitées:
```
x_lissé[t] = α·x_mesuré[t] + (1-α)·x_lissé[t-1]
```
où α ∈ [0,1] contrôle réactivité vs fluidité.

## 🛠️ Technologies
- **MuJoCo 3.0**: Simulation physique rapide avec dynamique de contact
- **NumPy**: Algèbre linéaire et calcul numérique
- **OpenCV**: Vision par ordinateur et visualisation
- **Python 3.10**: Langage moderne compatible async

## 📈 Extensions Possibles
- [ ] Filtre de Kalman pour une meilleure estimation d'état
- [ ] Optimisation de trajectoire (MPC, iLQR)
- [ ] Suivi multi-cibles
- [ ] Déploiement sur matériel réel (Franka Panda)
- [ ] IK basé sur l'apprentissage (approximateur par réseau de neurones)
- [ ] Évitement d'obstacles

## 🐛 Dépannage

**Le robot bouge trop vite:**
```python
# Réduire step_size et dq_limit dans ik_control.py
```

**Mouvement saccadé:**
```python
# Augmenter smooth_alpha dans main.py (plus de lissage)
```

**IK ne converge pas:**
```python
# Augmenter damping_base ou iterations dans IKConfig
```