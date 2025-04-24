# 🐍 SnakeGame AI – Deep Q-Learning avec Obstacles Dynamiques

Un projet de jeu Snake contrôlé par une intelligence artificielle utilisant le Deep Q-Learning (DQN). Le serpent apprend à survivre, à manger, et à éviter des obstacles générés aléatoirement sur le terrain.

---

## 📌 Objectifs du projet

- Implémenter une **IA basée sur l'apprentissage par renforcement** pour le jeu Snake.
- Ajouter une **génération dynamique d'obstacles** : carrés et rectangles placés aléatoirement à chaque partie.
- Empêcher l'IA de rester coincée dans des **boucles de mouvement**.
- Tester la **robustesse du modèle** dans des environnements variés.

---

## 🧠 Technologies utilisées

- `Python` + `Pygame` : pour créer l’environnement de jeu.
- `PyTorch` : pour le modèle de Deep Q-Learning.
- `Numpy` : pour les manipulations de données.
- `Matplotlib` (optionnel) : pour la visualisation des performances.

---

## 🎮 Fonctionnement

### ⚙️ 1. Environnement (`game.py`)
- Génère un terrain avec :
  - 3 **carrés** (taille fixe)  
  - 2 **rectangles** (taille fixe)  
- Empêche la nourriture et les obstacles de se chevaucher ou d’apparaître au centre.

### 🧠 2. Modèle d'IA (`model.py`)
- Réseau simple : `Linear_QNet` (entrée 11 → 256 → sortie 3).
- Trois actions possibles : `gauche`, `droite`, `tout droit`.

### 🔁 3. Entraînement (`train.py`)
- Apprentissage par renforcement basé sur la fonction de Bellman.
- Récompense +10 pour manger, -10 pour collision, -0.1 par pas pour encourager l'efficacité.

### 🧪 4. Évaluation (`testModel.py`)
- L’IA joue automatiquement plusieurs parties et affiche le score.
- Intègre une **composante aléatoire** dans les mouvements (ex: 10%) pour améliorer l’exploration.

---

## 🧠 Représentation de l'état (entrée du modèle)

| État | Description |
|------|-------------|
| Danger devant, gauche, droite | Booléens |
| Direction actuelle | 4 directions encodées |
| Position de la nourriture | Haut / Bas / Gauche / Droite par rapport à la tête |

---

## 💡 Améliorations intégrées

✅ Boucles détectées et pénalisées  
✅ Mouvement alternatif intelligent si collision imminente  
✅ Obstacles aléatoires à chaque partie  
✅ Probabilité configurable de mouvements aléatoires

---

## 📈 Résultats

- L’agent apprend efficacement dans un environnement sans et avec obstacles.
- Le modèle reste robuste même avec des obstacles dynamiques.
- Réduction significative des parties bloquées en boucle.

---

## 🚀 Lancer le projet

```bash
git clone https://github.com/ton-pseudo/SnakeGame-AI.git
cd SnakeGame-AI
pip install -r requirements.txt
python agent3.py     # Pour entraîner le modèle
python testModel2.py # Pour tester un modèle existant
