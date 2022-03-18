# -*- coding: utf-8 -*-
import data_preprocessing as dp
import matplotlib.pyplot as plt
import numpy as np
from random import randint
from math import sqrt
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, silhouette_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.exceptions import ConvergenceWarning
from warnings import filterwarnings
filterwarnings(action="ignore", category=ConvergenceWarning)

def draw_cv_results(title, cv_models):
    """
    Disegna i grafici per i vari split di cross validation, solo per modelli GridSearchCV

    """
    cvm = [x for x in cv_models if type(x) is GridSearchCV]
    if len(cvm) == 0:
        return
    cv = randfcv.cv if randfcv.cv != None else 5
    fig, axs = plt.subplots(len(cvm), figsize=(10, 6*len(cvm)))
    fig.suptitle(title)
    if len(cvm) == 1:
        axs = [axs]
    for ax, model in zip(axs, cvm):
        results = model.cv_results_
        ax.set_title(model.estimator.__class__.__name__)
        ax.set_ylabel("Score")
        ax.set_xlabel("CV splits")
        ax.set_xticks(range(cv))
        for i, m in enumerate(results["params"]):
            y = list()
            for j in range(cv):
                y.append(results["split{}_test_score".format(j)][i])
            p = ax.plot(range(cv), y, label=str(m).replace("{", "").replace("}", ""), marker="o")
            ax.plot(range(cv), [results["mean_test_score"][i]]*cv,
                    linestyle="--", color=p[0].get_color())
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
def draw_predicts(title, models, test_X, test_y):
    fig, axs = plt.subplots(len(models), figsize=(7, 7*len(models)))
    fig.suptitle(title)
    if len(models) == 1:
        axs = [axs]
    for ax, model in zip(axs, models):
        preds = model.predict(test_X)
        p = max(max(preds), max(test_y)), min(min(preds), min(test_y))
        ax.set_title(model.estimator.__class__.__name__ 
                     if type(model) is GridSearchCV else model.__class__.__name__)
        ax.set_ylabel("Predictions")
        ax.set_xlabel("Values")
        ax.scatter(test_y, preds, color='crimson')
        ax.plot(p, p, 'b-')

def eval_model(models, test_X, test_y):
    maes = np.zeros(len(models))
    for i, model in enumerate(models):
        name = ""
        if type(model) is GridSearchCV:
            name = "("+model.estimator.__class__.__name__+")"
        print("Valutazione di", model.__class__.__name__, name)
        score = model.score(test_X, test_y)
        rmse = sqrt(mean_squared_error(test_y, model.predict(test_X)))
        mae = mean_absolute_error(test_y, model.predict(test_X))
        print("R2 score   = ", score)
        print("RMSE score = ", rmse)
        print("MAE score  = ", mae)
        maes[i] = mae
    return maes

def train_models(models, train_X, train_y):
    for model in models:
        name = ""
        if type(model) is GridSearchCV:
            name = "("+model.estimator.__class__.__name__+")"
        print("Training", model.__class__.__name__, name)
        model.fit(train_X, train_y)

def train_eval(models, train, test, preproc, draw=False, title=""):
    train_X = preproc.fit_transform(train.drop(columns="price"))
    train_y = train.price
    test_X = preproc.transform(test.drop(columns="price"))
    test_y = test.price
    
    print("-"*10+title+"-"*10)
    train_models(models, train_X, train_y)
    eval_model(models, test_X, test_y)
    print("-"*(20+len(title)))
    if draw:
        draw_cv_results(title+" - CV", models)
        draw_predicts(title+" - Predictions",models, test_X, test_y)
        
    return train_X # TODO: DA TOGLIERE
    
def train_eval_cluster(models, train, test, preproc, k, models_per_cl=False, 
                       latlong=True, draw=False):
    """preproc_kmeans = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
        ("passthrough", make_column_selector(dtype_include="number"))
    )
    preproc.fit(train.drop(columns="price"))
    train_X = preproc_kmeans.fit_transform(train.drop(columns="price"))
    test_X = preproc_kmeans.transform(test.drop(columns="price"))"""
    if latlong:
        train_X = train.loc[:, ["lat","long"]]
        test_X = test.loc[:, ["lat","long"]]
    else:
        train_X = preproc.fit_transform(train.drop(columns="price"))
        test_X = preproc.transform(test.drop(columns="price"))

    # TRAINING
    kmeans = KMeans(n_clusters=k, algorithm="full")
    kmeans.fit(train_X)
    train["cluster"] = kmeans.labels_
    cl_train_X = [preproc.transform(x.drop(columns=["price", "cluster"]))
                  for _, x in train.groupby(train["cluster"])]
    cl_train_y = [x.price for _, x in train.groupby(train["cluster"])]
    # TESTING
    test["cluster"] = kmeans.predict(test_X)
    cl_test_X = [preproc.transform(x.drop(columns=["price", "cluster"]))
                 for _, x in test.groupby(test["cluster"])]
    cl_test_y = [x.price for _, x in test.groupby(test["cluster"])]
    if models_per_cl:
        cl_models = [[clone(x)] for x in models]
        err = np.zeros(1)
    else:
        cl_models = [[clone(x) for x in models] for i in range(k)]
        err = np.zeros(len(models))
    if draw:
        fig, axs = plt.subplots(ncols=2, figsize=(16,4))
        axs[0].set_title("Training data clusters")
        axs[1].set_title("Testing data clusters")
        for i, d in enumerate((train, test)):
            axs[i].set_ylabel("Price")
            axs[i].set_xlabel("Sqft_living")
            axs[i].scatter(d.sqft_living, d.price, c=d.cluster)
    for i in range(k):
        print("TRAINING CLUSTER", i)
        print("{}/{}".format(cl_test_X[i].shape[0], len(test)))
        train_models(cl_models[i], cl_train_X[i], cl_train_y[i])
        print("VALUTAZIONE CLUSTER", i)
        err += eval_model(cl_models[i], cl_test_X[i], cl_test_y[i]) * cl_test_X[i].shape[0]
    print("MAE score congiunto dei cluster = ", err/len(test))
    
def kmeans_choice(data, preproc=None, max_k=10):
    err = list()
    sil = list()
    fig, axs = plt.subplots(ncols=2, figsize=(10,4))
    # Per testare il kmeans solo su latitudine e longitudine
    # data = data.loc[:, ["lat", "long"]]
    if preproc != None:
        prep_data = preproc.fit_transform(data)
    else:
        prep_data = data
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, algorithm="full")
        kmeans.fit(prep_data)
        if k != 1:
            sil.append(silhouette_score(prep_data, kmeans.labels_))
        err.append(kmeans.inertia_)
    for ax in axs:
        ax.set_xlabel("K value")
    axs[0].set_ylabel("Sum of errors")
    axs[0].plot(range(1, max_k + 1), err, marker="o")
    axs[1].set_ylabel("Silhouette score")
    axs[1].plot(range(2, max_k + 1), sil, marker="o")

if __name__ == "__main__":
    data = dp.Dataset("../data/data.csv")
    
    #seed = randint(0, 100_000)
    seed = 19751
    cvsplits = 5
    print("Usato il seed:", seed)
    
    preproc = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
        (StandardScaler(), make_column_selector(dtype_include="number"))
    )
    
    # APPRENDIMENTO SUPERVISIONATO
    # Neural network
    nn = MLPRegressor(max_iter=300, random_state=0)
    #grid = {"alpha": [0.0001, 0.001, 0.01, 0.1]}
    grid = [{"solver": ["lbfgs"],
             "hidden_layer_sizes": [(5,), (10,), (10, 5)],
             "alpha": [0.0001, 0.001, 0.01]},
            {"solver": ["adam"],
             "hidden_layer_sizes": [(10,), (10, 5), (20, 10)],
             "learning_rate_init": [0.01, 0.1]}]
    nncv = GridSearchCV(nn, grid, cv=cvsplits)
    # Lasso SGD
    sgdlasso = SGDRegressor(penalty="l1", tol=None, alpha=0.001, random_state=0)
    grid = {"learning_rate": ["invscaling", "adaptive"], "eta0": [0.03, 0.01, 0.001]}
    sgdlassocv = GridSearchCV(sgdlasso, grid, cv=cvsplits)
    # Ridge SGD
    sgdridge = SGDRegressor(penalty="l2", tol=None, alpha=0.001, random_state=0)
    grid = {"learning_rate": ["invscaling", "adaptive"], "eta0": [0.03, 0.01, 0.001],
             "alpha": [0.001, 0.01]}
    sgdridgecv = GridSearchCV(sgdridge, grid, cv=cvsplits)
    # Decision Tree
    tree = DecisionTreeRegressor(random_state=0)
    grid = {"max_depth": [5, 10], "min_samples_leaf": range(1, 10, 2)}
    treecv = GridSearchCV(tree, grid, cv=cvsplits)
    # Ensemble
    randf = RandomForestRegressor(n_estimators=100, random_state=0)
    grid = {"max_features": ["auto", "sqrt", "log2"],
            "max_depth": [None, 5, 10]}
    randfcv = GridSearchCV(randf, grid, cv=cvsplits)
    # Suppor vector machine
    svm = SVR()
    grid = [{"kernel": ["linear", "rbf", "sigmoid"],
             "C": [1000, 10_000, 50_000],
             "epsilon": [0.1, 0.2]},
            {"kernel": ["poly"],
             "degree": [2,3,4]}]
    svmcv = GridSearchCV(svm, grid, cv=cvsplits)
    
    #models = [nncv, sgdlassocv, sgdridgecv, treecv, randfcv, svmcv]
    models = [randfcv, svmcv]
    #kmeans_choice(dp._data_preprocess(data.extended_data), preproc)
    
    """# Splitting del dataset normale, dataset originale
    a1=train_eval(models, 
                  *dp.train_test_equal_split(data.original_data, random_state=seed),
                  preproc, title="ORIGINAL DATA", draw=False)
    models_nocv = [x if type(x) != GridSearchCV else x.best_estimator_ for x in models]
    # Splitting del dataset normale, dataset esteso
    a2=train_eval(models_nocv,
                  *dp.train_test_equal_split(data.extended_data, ext=True, random_state=seed),
                  preproc, title="EXTENDED DATA")
    # Splitting del dataset con città differenti, dataset originale
    a3=train_eval(models_nocv,
                  *dp.train_test_diffcities_split(data.original_data, random_state=seed),
                  preproc, title="ORIGINAL DATA / DIFFERENT CITIES SPLITTING")
    # Splitting del dataset con città differenti, dataset esteso"""
    a4=train_eval(models,
                  *dp.train_test_diffcities_split(data.extended_data, ext=True, random_state=seed),
                  preproc, title="EXTENDED DATA / DIFFERENT CITIES SPLITTING")
    models_nocv = [x if type(x) != GridSearchCV else x.best_estimator_ for x in models]
    # Random forest sul primo cluster e SVR sul secondo
    # KMeans
    train_eval_cluster(models_nocv,
                       *dp.train_test_diffcities_split(data.extended_data, ext=True, random_state=seed),
                       preproc, k=3, models_per_cl=False, draw=True)