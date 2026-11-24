# 🧠 Devoir Projet 1 : Bases de PyTorch

Ce projet présente l'apprentissage des bases de **PyTorch** et la résolution complète de 4 exercices pratiques de Machine Learning et Deep Learning.

---

## 📂 Contenu du Dossier

- **`Activite_pratique_Bases_de_Pytorch.ipynb`** : Le notebook complet contenant le cours et les 4 exercices résolus et pré-exécutés avec tous leurs graphiques.
- **`models/`** : Dossier contenant les poids des modèles entraînés sauvegardés (`mlp_mnist.pth`, `mnist_cnn.pth`).
- **`.gitignore`** : Fichier ignorant le cache de données MNIST et l'environnement local.

---

## 📝 Description des Exercices

### 1. 💼 Exercice 1 : Régression Linéaire Multiple
- **Objectif** : Prédire le salaire mensuel en dirhams (DH) selon l'ancienneté (années) et le niveau d'expertise (1 à 5).
- **Modèle** : Un neurone linéaire `nn.Linear(2, 1)` entraîné avec la perte MSE.
- **Résultat** : Erreur moyenne (MAE) de ~392 DH et prédiction précise pour un profil à 7 ans d'ancienneté.

### 2. 🌸 Exercice 2 : Classification des Fleurs Iris
- **Objectif** : Classifier les fleurs en 3 espèces (*setosa*, *versicolor*, *virginica*) à partir de 4 mesures.
- **Modèle** : Réseau de neurones à une couche cachée `Linear(4, 16) -> ReLU -> Linear(16, 3)`.
- **Résultat** : 100% d'accuracy sur le jeu de test et matrice de confusion sans erreur.

### 3. 🔢 Exercice 3 : Classification MNIST avec MLP
- **Objectif** : Reconnaître des chiffres manuscrits (0 à 9) avec un perceptron multicouche.
- **Modèle** : Aplatissement de l'image $28 \times 28$ suivi de deux couches denses (`784 -> 128 -> 10`).
- **Résultat** : 96.5% d'accuracy sur le jeu de test et sauvegarde des poids dans `mlp_mnist.pth`.

### 4. 🖼️ Exercice 4 : Classification MNIST avec CNN
- **Objectif** : Améliorer la reconnaissance des chiffres grâce aux convolutions spatiales.
- **Modèle** : 2 blocs de convolution et pooling (`Conv2d`, `MaxPool2d`) avec régularisation `Dropout(0.25)`.
- **Résultat** : Plus de 98.8% d'accuracy et affichage des chiffres difficiles mal classés.

---

## 🚀 Comment Exécuter

1. Installez les dépendances :
   ```bash
   pip install torch torchvision matplotlib scikit-learn
   ```
2. Ouvrez le notebook `Activite_pratique_Bases_de_Pytorch.ipynb` dans VS Code ou Jupyter.
3. Toutes les cellules sont déjà pré-exécutées avec leurs graphiques et métriques visibles.
