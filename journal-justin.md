# Journal  
__Auteur:__ Justin Charbonneau  
__Mots Cles:__ XGBoost, Hyperopt, Backtesting, Entity Embeddings, Target Encoding, Regression

## Jour 1

**Apprentissage:**

- [x] Ecouter tutoriels youtube sur @adaboost, @gradientboost and @xgboost.
- [x] Création d'un environement yml pour le projet.
- [x] Passer au travers du notebook d'exploration fait par FastAI. On realise qu'il y a des valeurs catégoriques et numériques, dont certains ont des valeurs nulles.

## Jour 2

- [x] Regarder au nombre de magasins par journée. Si on prédit à chaque jour pour tous les magasins, alors on peut utiliser un split temporel pour du 'backtesting'. Ceci est fait avec la fonction `TimeSeriesSplit` de sklearn.
- [x] Déterminer méthodologie et faire le diagramme. Voir figure 1.
- [x] La competition de Rossman sur Kaggle demande l'utilisation de la 'Root Mean Squared Percentage Error' (RMSPE). L'algorihtme XGBoost ne vient pas avec cette loss pré-défini, alors il faut la spécifier. Pour ce faire, j'ai trouver une personne qui a trouver comment le faire dans le forum: [Chenglong Chen](https://www.kaggle.com/c/rossmann-store-sales/discussion/16794). Voir Figure 2.
- [x] On roule XGBoost 

**Figure 1:** Méthodologie de 'backtesting'. Lors de l'optimization des hyperparamètres, je saute les premières itérations et exécute l'entraînement et validation pour les trois derniers 'folds'. Sinon, ça prendrait beaucoup trop de temps.
![image](https://user-images.githubusercontent.com/25487881/78314966-a32d8600-7529-11ea-9560-b80d5c1e5435.png)

**Figure 2:** Définir la fonction de la loss (RMSPE)
Source: [Chenglong Chen](https://www.kaggle.com/c/rossmann-store-sales/discussion/16794)
```python
def ToWeight(y):
    w = np.zeros(y.shape, dtype=float)
    ind = y != 0
    w[ind] = 1./(y[ind]**2)
    return w

def rmspe(yhat, y):
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean( w * (y - yhat)**2 ))
    return rmspe

def rmspe_xg(yhat, y):
    y = y.get_label()
    y = np.exp(y) - 1
    yhat = np.exp(yhat) - 1
    w = ToWeight(y)
    rmspe = np.sqrt(np.mean(w * (y - yhat)**2))
    return "rmspe", rmspe
```

## Jour 3

- [x] Lire sur différentes methodes de recherche d'hyperparametre (grid search, random search, bayesian optimisation). Lire sur Hyperopt qui a été utilisé dans des projets au CRIM. Il semblait être bien fait et simple à utiliser. -> [Hyperopt](https://github.com/hyperopt/hyperopt). En bref, pour utiliser Hyperopt, il faut définir deux fonctions. Voir la figure 3 dans le jour 4 pour une illustration des résultats.

La focntion 'optimize' défini l'espace de recherche. On utilise fmin qui cherche à minimiser le score puis on a spécifier l'algorithme de recherche `tpe.suggest`. Finalement, le nombre d'évaluations est de 100. La variable trials est appelé afin de sauver les expériences et résultats de chaque évaluation. 
```python
def optimize():
    space = {'eta': hp.quniform('eta', 0.01, 0.3, 0.001),
             'max_depth':  hp.choice('max_depth', np.arange(5, 10, dtype=int)),
             'gamma': hp.quniform('gamma', 0, 5, 0.5)}
             
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=100)
    
    return best
    
def score(params):
    list_of_scores = []
    Do Backtesting with TimeSeriesSplit
        1) Split data according to index returned by TimeSeriesSplit
        2) Train XGBoost with params and record RMSPE score
        list_of_scores.append(score)
        
    return(np.mean(score))
    
# trials will contain logging information
trials = Trials()

best_hyperparams = optimize()
``` 

- [x] Rouler XGBoost avec Hyperopt 
- [x] Lire sur le papier original [Entity Embeddings (EE)](https://arxiv.org/pdf/1604.06737.pdf)
- [x] Trouver du code en Pytorch pour la création des EEs. Ce blog était bon! -> [blog post](https://yashuseth.blog/2018/07/22/pytorch-neural-network-for-tabular-data-with-categorical-embeddings/)
- [x] Comparaison avec EE de fastai. FastAI appliques le Batch Norm (BN) et le Dropout avant l'activation ReLU. Dans le blog, on l'applique après. Une meilleure compréhension sur ceci serait nécessaire, mais après avoir roulé les deux, j'ai poursuivi avec BN et dropout après ReLU.
- [x] Adapter le code pour ne pas qu'il traite les donnes numeriques, seulement categorique. Pas necessaire, mais ca me permet d'apprendre PyTorch
- [x] Faire une fonction qui va chercher les poids (embeddings) respectifs et remplacer les valeurs catégoriques par les embeddings

**Figure 3:** Architecture pour deux colones avec valeurs categoriques
![image](https://user-images.githubusercontent.com/25487881/78181963-42bc1d00-7433-11ea-8236-6dd6f64e247a.png)

## Jour 4

- [x] Rajouter une classification pour les données nulls en tant que #NAN#
- [x] Trouver pourquoi mes résultats dans ma recherche d'hyperparamètres ne sont pas pareils après quand j'entraîne le modèle avec les meilleurs hyperparamètres.
- [ ] Comprendre comment choisir le bon learning rate. (Pour l'instant, j'ai pu constaté qu'il est bonne pratique d'illustrer la loss pour voir quel effet un learning rate de 10 vs 100 aurait. Dans ce cas, il y avait un plateau au début de l'entraînement, qu'il fallait dépasser avant que le modèle puisse continuer à apprendre.  Ceci avait un effet négatif dans la recherche d'hyperparamètre pour le 'eta', car si je met un 'early stopping' après 10 iterations, l'algo restait pogner dans le plateau et arrêtait quand l' 'eta' était petit. Donc, après la recherche, ça disait qu'un 'eta' de 0.2 etait le meilleure. Ceci allait allencontre de la majorité des notebooks sur kaggle qui utilisaient des learning rate d'environ 0.025.
- [x] Rouler la recherche d'hyperparamètres pendant la nuit. 

Notes sur Hyperopt:
> - hp.choice retourne un index. Il est possible d'utiliser la fonction `space_eval` de hyperopt pour faire le switch. (Ex: `best_hyperparams = space_eval(space, best_hyperparams)`

**Figure 4:** Valeurs recherchées lors de la recherche d'hyperparamètre pour l'expérience 002. Comme mentioné, les valeurs pour `max_depth` ne sont pas les vraies valeurs, mais l'index qui map {0:5, 1:6, 2:7, 3:8, 4:9}
![image](https://user-images.githubusercontent.com/25487881/78713644-98d40900-78e8-11ea-9c54-1e961d97c11b.png)

## Jour 5

- [x] Soumission d'une Pull Request 
- [ ] Réaliser que clean_train_valid peut être combiné avec clean_internal_test pour avoir un plus gros dataset. Ainsi, faut re-rouler le tout.
- [ ] Faire l'algo [@gradientboost](https://www.youtube.com/watch?v=2xudPOBz-vs&t=281s) à la main pour me familiariser.
- [ ] Re-rouler les résultats et soummettres les predictions sur Kaggle maintenant que les hyperparametres ont ete trouver.

# TODO

- [ ] Update environment yml
- [ ] Faire un setup avec MLFlow

Reflexion a avoir:

- [ ] Comment les variables nulls sont traités pour des donnes continues
- [ ] Comment les variables nulls sont traités pour des donnes catégoriques
- [ ] Est-ce que normaliser les données va changer comment le split des histogrammes est fait?

# Resultats

| Experiment ID | Categorical Variables | NaN-cats | NaN-cont | Target Transformation | Hyperparameter Search | Backtesting            | Private Score | Public Score
|---------------|-----------------------|----------|----------|-----------------------|-----------------------|------------------------|---------------|--------------
| 001           | Target encoder        | XGBoost  | XGBoost  | Log transform         | Default               | No                     | 0.16925       | 0.17975
| 002           | Target encoder        | XGBoost  | XGBoost  | Log transform         | HyperOpt (100)        | TimeSeriesSplit k = 3  | 0.13975       | 0.12481
| 003           | Entity Embeddings     | #NAN#    | FastAI   | Log transform         | Default               | No                     | 0.15251       | 0.14079
| 004           | Entity Embeddings     | #NAN#    | FastAI   | Log transform         | HyperOpt (100)        | TimeSeriesSplit k = 3  | 0.13081       | 0.11572
