# 🧠 Devoir Projet 1 : Fondamentaux de Deep Learning avec PyTorch

Bienvenue dans le premier devoir pratique consacré aux **fondations de PyTorch** et aux architectures fondamentales de Deep Learning ! 🚀

Ce projet couvre les concepts essentiels du framework (tenseurs, calcul différentiel automatique `Autograd`, boucles d'apprentissage) et propose la **résolution complète et détaillée de 4 cas pratiques** allant de la régression linéaire aux réseaux de neurones convolutifs (CNN).

---

## 📌 Sommaire
- [🎯 Objectifs de l'activité](#-objectifs-de-lactivité)
- [📂 Structure du projet](#-structure-du-projet)
- [🧪 Synthèse des 4 Exercices Résolus](#-synthèse-des-4-exercices-résolus)
  - [1. Régression Multiple — Prédiction des Salaires](#1-régression-multiple--prédiction-des-salaires)
  - [2. Classification Multiclasse — Dataset Iris 🌸](#2-classification-multiclasse--dataset-iris-)
  - [3. Perceptron Multicouche (MLP) sur MNIST 🔢](#3-perceptron-multicouche-mlp-sur-mnist-)
  - [4. Réseau Convolutif (CNN) sur MNIST 🖼️](#4-réseau-convolutif-cnn-sur-mnist-️)
- [📊 Résultats & Comparaisons](#-résultats--comparaisons)
- [💻 Guide d'Exécution](#-guide-dexécution)

---

## 🎯 Objectifs de l'activité
- Maîtriser la manipulation des **Tenseurs PyTorch** et l'accélération matérielle (CPU / GPU CUDA).
- Comprendre le graphe de calcul dynamique et la rétropropagation grâce à **`autograd`** (`backward()`, `grad`).
- Structurer des modèles modulaires avec `nn.Module` et `nn.Sequential`.
- Maîtriser le cycle d'apprentissage standard : `optimizer.zero_grad()`, calcul de loss, `loss.backward()`, `optimizer.step()`.
- Implémenter et évaluer des architectures adaptées aux données tabulaires et aux images.

---

## 📂 Structure du projet
```bash
devoir projet 1/
├── Activite_pratique_Bases_de_Pytorch.ipynb  # Notebook Jupyter complet & commenté
├── README.md                                 # Guide de démonstration & synthèse
└── models/                                   # Poids des modèles sauvegardés (générés après run)
    ├── mlp_mnist.pth
    └── mnist_cnn.pth
```

---

## 🧪 Synthèse des 4 Exercices Résolus

### 1. Régression Multiple — Prédiction des Salaires 💼
- **Problématique** : Prédire le salaire mensuel (en Dirhams) d'un salarié à partir de son **ancienneté** ($x_1$) et de son **niveau d'expertise** ($x_2$).
- **Modélisation** :
  $$\text{Salaire} = w_1 \cdot \text{Ancienneté} + w_2 \cdot \text{Expertise} + b$$
- **Points clés d'implémentation** :
  - Standardisation des variables prédictives avec les statistiques du jeu d'entraînement.
  - Normalisation de la cible en milliers de DH pour stabiliser la descente de gradient.
  - Entraînement avec `nn.MSELoss()` et l'optimiseur `SGD(lr=0.05)`.
- **Validation** :
  - Test sur un profil cible : **7 ans d'ancienneté**, **expertise 4/5**.
  - Formule théorique sans bruit : $4000 + 600 \times 7 + 1500 \times 4 = 14\,200\text{ DH}$.
  - Le modèle converge fidèlement vers cette valeur avec une erreur moyenne (MAE) minime (~350-400 DH).

---

### 2. Classification Multiclasse — Dataset Iris 🌸
- **Jeu de données** : 150 échantillons de 3 espèces d'iris (*Setosa*, *Versicolor*, *Virginica*) décrits par 4 caractéristiques morphologiques.
- **Architecture Réseau** :
  ```
  Entrée (4) ──> [Linear(4 -> 16)] ──> [ReLU] ──> [Linear(16 -> 3)] ──> Logits
  ```
- **Réponses aux questions clés** :
  - **Pourquoi `torch.long` ?** `nn.CrossEntropyLoss` exige des étiquettes entières sous forme de tenseur 64 bits (`int64`) pour indexer directement les classes de coût.
  - **Pourquoi ne pas faire `fit` du scaler sur le test ?** Pour proscrire formellement la *fuite d'information* (*data leakage*) depuis les données futures.
- **Résultats** :
  - Accuracy de test : **96.7% à 100%**.
  - Comparaison : Le MLP avec couche cachée sépare nettement mieux les classes non linéaires (*Versicolor* vs *Virginica*) qu'un simple modèle linéaire direct `Linear(4, 3)`.

---

### 3. Perceptron Multicouche (MLP) sur MNIST 🔢
- **Jeu de données** : 70 000 images de chiffres manuscrits ($28 \times 28$ pixels en niveaux de gris).
- **Architecture** :
  - Aplatissement : `nn.Flatten()` (dimension $28 \times 28 = 784$).
  - Couche dense cachée : `nn.Linear(784, 128)` + activation `ReLU`.
  - Couche de sortie : `nn.Linear(128, 10)` (scores logits pour chaque chiffre).
- **Métriques complètes sur le jeu de test** :
  - **Accuracy** : $\approx 97.4\%$
  - **Précision (Macro)** : $\approx 0.974$
  - **Rappel (Macro)** : $\approx 0.973$
  - **F1-Score (Macro)** : $\approx 0.973$
- **Persistance** : Sauvegarde propre du dictionnaire d'états `model.state_dict()` dans `mlp_mnist.pth` et rechargement sans perte de poids.

---

### 4. Réseau Convolutif (CNN) sur MNIST 🖼️
- **Architecture Convolutive** :
  ```mermaid
  flowchart LR
      In["Image 1x28x28"] --> Conv1["Conv2d(1->16, k=3, p=1) + ReLU"]
      Conv1 --> Pool1["MaxPool2d(2) -> 16x14x14"]
      Pool1 --> Conv2["Conv2d(16->32, k=3, p=1) + ReLU"]
      Conv2 --> Pool2["MaxPool2d(2) -> 32x7x7"]
      Pool2 --> Flat["Flatten (1568)"]
      Flat --> Dense1["Linear(1568 -> 128) + ReLU + Dropout(0.25)"]
      Dense1 --> Out["Linear(128 -> 10)"]
  ```
- **Pourquoi la couche linéaire reçoit-elle $32 \times 7 \times 7$ valeurs ?**
  L'image initiale $28 \times 28$ subit deux sous-échantillonnages par max-pooling successifs de facteur 2 :
  $$28 \xrightarrow{/2} 14 \xrightarrow{/2} 7$$
  Comme le 2e bloc convolutif extrait 32 canaux de caractéristiques, le volume spatial final avant classification est de dimension $(32, 7, 7)$, soit un vecteur plat de $32 \times 7 \times 7 = 1568$ valeurs.
- **Résultats obtenus** :
  - Accuracy de test : **$\ge 98.9\%$** (gain net de précision par rapport au MLP classique grâce à l'invariance par translation des filtres convolutifs).
  - Visualisation des erreurs : analyse des chiffres ambigus (ex: un 4 écrit comme un 9, ou un 7 ressemblant à un 1).

---

## 📊 Résultats & Comparaisons

| Modèle | Domaine | Paramètres principaux | Précision / Métrique |
| :--- | :--- | :--- | :--- |
| **Régression Multiple** | Salaires RH | 2 poids, 1 biais | MAE $\approx 370\text{ DH}$ |
| **MLP Iris** | Botanique | 4 $\to$ 16 $\to$ 3 | Accuracy $\approx 96.7\% - 100\%$ |
| **MLP MNIST** | Vision (Chiffres) | 784 $\to$ 128 $\to$ 10 | Accuracy $\approx 97.4\%$ |
| **CNN MNIST** | Vision (Chiffres) | 2 blocs Conv/Pool + Dropout | **Accuracy $\approx 99.1\%$** |

---

## 💻 Guide d'Exécution

### Option 1 : Dans VS Code en local
1. Créez et activez votre environnement virtuel Python :
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # Windows
   ```
2. Installez les dépendances :
   ```bash
   pip install torch torchvision matplotlib scikit-learn
   ```
3. Ouvrez `Activite_pratique_Bases_de_Pytorch.ipynb` et exécutez les cellules séquentiellement !

### Option 2 : Sur Google Colab
Glissez simplement le fichier `Activite_pratique_Bases_de_Pytorch.ipynb` sur [Google Colab](https://colab.research.google.com). L'environnement GPU gratuit exécutera l'entraînement du CNN en quelques secondes. ✨
