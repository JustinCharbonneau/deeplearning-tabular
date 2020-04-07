# Day 1

- [x] Ecouter tutoriels youtube sur @adaboost, @gradientboost and @xgboost.
- [x] Creation d'un environment yml pour le projet
- [x] Passer au travers du notebook d'exploration

# Day 2

- [x] Regarder aux nombres de magasins par journee. Si on predit a chaque jour pour tous les magasins, alors on peut utiliser un split temporelle pour du backtesting
- [x] Determiner methodologie et faire le diagrame *Figure 1*
**Figure 1:** Methodologie de backtesting. Lors de l'otimization des hyperparametre, je skip les premieres iterations et execute l'entraintement
et validation pour les trois dernier folds. Sinon, ca prendrais beaucoup trop de temps.
- [x] On roule XGBoost 

**Figure 1:** Methodologie de backtesting. Lors de l'otimization des hyperparametre, je skip les premieres iterations et execute l'entraintement
et validation pour les trois dernier folds. Sinon, ca prendrais beaucoup trop de temps.
![image](https://user-images.githubusercontent.com/25487881/78314966-a32d8600-7529-11ea-9560-b80d5c1e5435.png)

# Day 3

- [x] Lire sur differentes methodes de recherche d'hyperparametre (grid search, random search, bayesian optimisation). Lire sur Hyperopt
qui a ete utiliser dans des projets au CRIM. Il semblait etre bien fait et simple a utiliser. -> [Hyperopt](https://github.com/hyperopt/hyperopt)
- [x] Rouler XGBoost avec Hyperopt 
- [x] Lire sur le papier originale [Entity Embeddings](https://arxiv.org/pdf/1604.06737.pdf)
- [x] Trouver du code en Pytorch pour la creation des EE. Ce blog etait bon! -> [blog post](https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/)
- [x] Comparaison avec EE de fastai. FastAI appliques le Batch Norm et Dropout avant l'activation ReLU. Dans le blog, on l'applique apres. Une meilleure
comprehension sur ceci serait necessaire, mais apres avoir rouler les deux, j'ai poursuivi avec BN et Dropout apres ReLU.
- [x] Adapter le code pour ne pas qu'il traite les donnes numeriques, seulement categorique. Pas necessaire, mais ca me permet
d'apprendre PyTorch
- [x] Faire une fonction qui va chercher les poids (embeddings) respectifs et remplacer les valeurs categoriques par les embeddings

**Figure 2:** Architecture pour deux colones avec valeurs categoriques
![image](https://user-images.githubusercontent.com/25487881/78181963-42bc1d00-7433-11ea-8236-6dd6f64e247a.png)

# Day 4

- [x] Rajouter une classification pour les donnees nulls en tant que #NAN#
- [x] Trouver pourquoi mes resultats dans ma recherche d'hyperparametre n'est pas pareil apres quand j'entraine le modele avec les meilleurs 
parametres
- [ ] Comprendre comment choisir le bon learning rate. ( A date, je vois qu'il est bonne pratique d'illustrer la loss pour voir quel effet un
learning rate de 10 vs 100 auraient. Dans ce cas, il y avais un plateau au debut de l'entrainement, qu'il fallait depasser avant que l'algo
puisse continuer a apprendre.  Ceci avait un effet negatif dans la recherche d'hyperparametre pour le 'eta', car si je met un early stopping apres
10 iterations, l'algo restait pogner dans le plateau et arretait quand le 'eta' etait petit. Donc, apres la recherche, ca disait qu'un 'eta' de 
0.2 etait le meilleure. Ceci allait allencontre de la majorite des notebooks sur kaggle qui utilisaient des learning rate d'environ 0.025.

Notes sur Hyperopt:
> - hp.choice retourne un index. Il est possible d'utiliser la fonction `space_eval` de hyperopt pour faire le switch. (Ex: `best_hyperparams = space_eval(space, best_hyperparams)`

- [x] Rouler la recherche d'hyperparametres pendant la nuit. 


# Day 5

- [ ] Faire l'algo [@gradientboost](https://www.youtube.com/watch?v=2xudPOBz-vs&t=281s) a la main pour me familiariser.
- [ ] Re-rouler les resultats et soummettres les predictions sur Kaggle maintenant que les hyperparametres ont ete trouver.

# TODO

- [ ] Update environment yml
- [ ] Faire un setup avec MLFlow

Reflexion a avoir:

- [ ] Comment les variables nulls sont traite pour des donnes continues
- [ ] Comment les variables nulls sont traite pour des donnes categoriques
- [ ] Est-ce que normaliser les donnees va changer comment le split des histograme est fait?

# Resultats

| Experiment ID | Categorical Variables | NaN-cats | NaN-cont | Target Transformation | Hyperparameter Search | Backtesting            | Private Score | Public Score
|---------------|-----------------------|----------|----------|-----------------------|-----------------------|------------------------|---------------|--------------
| 001           | Target encoder        | XGBoost  | XGBoost  | Log transform         | Default               | No                     | 0.16925       | 0.17975
| 002           | Target encoder        | XGBoost  | XGBoost  | Log transform         | HyperOpt (100)        | TimeSeriesSplit k = 3  | 0.13975       | 0.12481
| 003           | Entity Embeddings     | #NAN#    | FastAI   | Log transform         | Default               | No                     | 0.15251       | 0.14079
| 004           | Entity Embeddings     | #NAN#    | FastAI   | Log transform         | HyperOpt (100)        | TimeSeriesSplit k = 3  | 0.13081       | 0.11572
