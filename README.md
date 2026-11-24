# Devoir Projet 1 : Bases de PyTorch

Ce projet presente l'apprentissage des bases de PyTorch et la resolution complete de 4 exercices pratiques de Machine Learning et Deep Learning.

---

## Contenu du Dossier

- **Activite_pratique_Bases_de_Pytorch.ipynb** : Notebook contenant les demonstrations du cours et les 4 exercices resolus et pre-executes.
- **models/** : Dossier contenant les poids des modeles sauvegardes (mlp_mnist.pth, mnist_cnn.pth).
- **.gitignore** : Fichier de configuration ignorant les caches temporaires et les donnees brutes.

---

## Description des Exercices

### 1. Exercice 1 : Regression Lineaire Multiple
- **Objectif** : Predire le salaire mensuel en dirhams (DH) selon l'anciennete et le niveau d'expertise.
- **Modele** : Neurone lineaire nn.Linear(2, 1) entraine avec la perte quadratique moyenne (MSE).
- **Resultat** : Erreur moyenne (MAE) d'environ 392 DH et convergence conforme a la formule theorique.

### 2. Exercice 2 : Classification des Fleurs Iris
- **Objectif** : Classifier les fleurs Iris en 3 especes distinctes a partir de 4 mesures morphologiques.
- **Modele** : Reseau de neurones a une couche cachee Linear(4, 16) -> ReLU -> Linear(16, 3).
- **Resultat** : 100% d'accuracy sur le jeu de test et matrice de confusion sans erreur.

### 3. Exercice 3 : Classification MNIST avec MLP
- **Objectif** : Reconnaitre les chiffres manuscrits de 0 a 9 a l'aide d'un perceptron multicouche.
- **Modele** : Aplatissement de l'image 28x28 suivi de deux couches denses (784 -> 128 -> 10).
- **Resultat** : 96.5% d'accuracy sur le jeu de test et sauvegarde des poids dans mlp_mnist.pth.

### 4. Exercice 4 : Classification MNIST avec CNN
- **Objectif** : Ameliorer la reconnaissance des chiffres par l'extraction de caracteristiques spatiales.
- **Modele** : Reseau convolutif a 2 blocs de convolution et pooling avec regularisation Dropout(0.25).
- **Resultat** : 98.8% d'accuracy sur le jeu de test et analyse des images mal classees.

---

## Comment Executer

1. Installer les dependances necessaires :
   ```bash
   pip install torch torchvision matplotlib scikit-learn
   ```
2. Ouvrir le notebook Activite_pratique_Bases_de_Pytorch.ipynb dans VS Code ou Jupyter.
3. Toutes les cellules sont deja pre-executees avec leurs sorties et graphiques visibles.
