# Détection de fake news en français par NLP et Machine Learning

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GOUBAGeorgette/detection-fake-news-nlp/blob/main/Untitled9.ipynb)

Projet de soutenance de Licence professionnelle en Génie logiciel (spécialité data), Université Virtuelle du Burkina Faso (UV-BF). **Mention Très Bien, 17/20.**

**Auteure :** Georgette GOUBA · [LinkedIn](https://linkedin.com/in/georgette-gouba-07279413d)

---

## Objectif

Construire un système qui classe une publication de réseau social ou d'actualité en **information vraie** ou **fake news**, comparer plusieurs modèles de Machine Learning, puis **expliquer pourquoi** le meilleur modèle prend ses décisions.

## Résultats en bref

Le meilleur modèle est le **Random Forest**, avec **95,26 % d'accuracy** et un **F1-score de 95,13 %** sur un jeu de test de 633 publications jamais vues pendant l'entraînement.

| Modèle | Accuracy | F1-score | Précision | Rappel | AUC-ROC |
|---|---|---|---|---|---|
| **Random Forest** | **95,26 %** | **95,13 %** | 95,13 % | 95,13 % | 98,04 % |
| XGBoost | 94,47 % | 94,35 % | 93,89 % | 94,81 % | 98,35 % |
| Régression logistique | 90,21 % | 90,06 % | 88,92 % | 91,23 % | 96,30 % |
| Naïve Bayes | 85,62 % | 85,39 % | 84,44 % | 86,36 % | 92,01 % |

**Stabilité (validation croisée stratifiée à 5 plis, F1-score moyen) :** Random Forest 0,940 (± 0,007), XGBoost 0,935 (± 0,005), Régression logistique 0,894 (± 0,008), Naïve Bayes 0,836 (± 0,013).

Matrice de confusion du Random Forest : il détecte aussi bien les vraies informations que les fake news (précision et rappel de 0,95 pour chaque classe).

![Matrice de confusion](images/matrice_confusion.png)
![Courbes ROC](images/courbe_roc.png)
![Comparaison des modèles](images/comparaison_modeles.png)
![Validation croisée](images/validation_croisee.png)

## Données

Le corpus final contient **3 162 publications en français** (1 622 vraies, 1 540 fake news), construit à partir de deux sources :

1. **Un jeu de données public Kaggle** de publications en français étiquetées vraies ou fausses : [nom du dataset et lien Kaggle à compléter].
2. **Des publications burkinabè collectées manuellement** (environ 300 publications après nettoyage), issues de médias en ligne (Burkina 24, Wakat Séra, LeFaso.net, RTB, Service d'Information du Gouvernement, Sidwaya, AIB, etc.) et d'un groupe WhatsApp, puis étiquetées à la main.

Avant l'entraînement : fusion des deux sources, suppression des lignes vides, **suppression des doublons** (5 050 → 3 162 publications) et mélange aléatoire.

> Les fichiers de données ne sont pas publiés dans ce dépôt (sources tierces et contenus issus des réseaux sociaux).

## Méthode

1. **Nettoyage du texte** : minuscules, suppression des URL, mentions, hashtags, chiffres et ponctuation, retrait des mots vides français (NLTK).
2. **Variables de style (7)** : nombre de mots, nombre de caractères, proportion de mots en majuscules, nombre de « ! » et de « ? », nombre de mots exagérés, nombre de mots négatifs.
3. **Vectorisation TF-IDF** : 2 000 variables (mots seuls et paires de mots, `min_df=2`, `sublinear_tf=True`), fusionnées avec les 7 variables de style, soit **2 007 variables** au total.
4. **Séparation** : 80 % entraînement (2 529 publications) et 20 % test (633), stratifiée, `random_state=42`.
5. **Modèles comparés** : Régression logistique, Naïve Bayes, Random Forest, XGBoost.
6. **Évaluation** : accuracy, F1-score, précision, rappel, AUC-ROC, matrice de confusion, courbes ROC, validation croisée stratifiée.
7. **Interprétabilité (SHAP)** : analyse des mots et des variables qui influencent le plus les décisions du Random Forest.
8. **Démonstration** : test sur des publications burkinabè jamais vues et petite interface web avec Gradio.

## Interprétation des décisions (SHAP)

[À compléter : indiquez ici 3 ou 4 observations tirées de votre graphique SHAP, par exemple les mots ou variables qui poussent le plus vers « fake » ou vers « vrai », et ce que cela vous apprend sur le style des fake news.]

## Limites

Je les indique volontairement : un score élevé ne veut pas dire que le modèle est prêt à l'emploi.

- **Un déséquilibre de sources** : environ 90 % du corpus vient du jeu de données Kaggle et seulement environ 10 % de publications burkinabè. Le modèle apprend donc surtout le style de ces données publiques, et ses performances sur l'actualité locale sont moins certaines.
- **Un test sur des publications locales encore limité** : sur trois publications burkinabè de démonstration, certaines confiances restent modestes (56 % et 71 % pour deux d'entre elles). Ces exemples illustrent la limite de généralisation ; ils ne constituent pas une évaluation.
- **Un étiquetage manuel** pour les données locales, donc susceptible d'erreurs.
- **Le français uniquement**, sans prise en compte des langues nationales ni des images ou vidéos.
- **Un outil d'aide, pas un juge** : il ne vérifie pas les faits, il repère des motifs statistiques.

## Pistes d'amélioration

- Collecter beaucoup plus de publications burkinabè pour rééquilibrer le corpus.
- Évaluer le modèle séparément sur les données locales.
- Essayer des modèles de langage pré-entraînés pour le français (par exemple CamemBERT).
- Ajuster le TF-IDF uniquement sur le jeu d'entraînement.
- Déployer la démonstration de façon permanente (par exemple sur Hugging Face Spaces).

## Technologies

Python · pandas · NumPy · scikit-learn · XGBoost · NLTK · SHAP · matplotlib · seaborn · Gradio · Google Colab

## Reproduire le projet

1. Ouvrez le notebook dans Google Colab avec le bouton en haut de cette page.
2. Placez vos fichiers de données dans l'environnement Colab : `kaggle_data_brut.csv` et `local_data_brut.csv` (séparateur `;`).
3. Exécutez les cellules dans l'ordre. La première installe les bibliothèques nécessaires.

## Contact

Georgette GOUBA · georgettegouba18@gmail.com
