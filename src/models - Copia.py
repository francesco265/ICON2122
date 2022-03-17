# -*- coding: utf-8 -*-
import data_preprocessing as dp
from random import randint
from math import sqrt
from sklearn.linear_model import SGDRegressor, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
import warnings
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings(action="ignore", category=ConvergenceWarning)

def draw_cv_results(title, cv_models):
    """
    Disegna i grafici per i vari split di cross validation, solo per modelli GridSearchCV

    """
    cvm = [x for x in cv_models if type(x) is GridSearchCV]
    fig, axs = plt.subplots(len(cvm), figsize=(10, 6*len(cvm)))
    fig.suptitle(title)
    if len(cvm) == 1:
        axs = [axs]
    for ax, model in zip(axs, cvm):
        if not type(model) is GridSearchCV:
            continue
        results = model.cv_results_
        ax.set_title(model.estimator.__class__.__name__)
        ax.set_ylabel("score")
        #ax.set_ylim(bottom=0)
        ax.set_xlabel("CV splits")
        ax.set_xticks(range(model.cv))
        for i, m in enumerate(results["params"]):
            y = list()
            for j in range(model.cv):
                y.append(results["split{}_test_score".format(j)][i])
            p = ax.plot(range(model.cv), y, label=str(m), marker="o")
            ax.plot(range(model.cv), [results["mean_test_score"][i]]*model.cv,
                    linestyle="--", color=p[0].get_color())
        ax.legend(loc="best")

def eval_model(model, preproc, X_test, y_test):
    name = ""
    if type(model) is GridSearchCV:
        name = "("+model.estimator.__class__.__name__+")"
    print("Valutazione di", model.__class__.__name__, name)
    model = make_pipeline(preproc, model)
    score = model.score(X_test, y_test)
    rmse = sqrt(mean_squared_error(y_test, model.predict(X_test)))
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print("R2 score   = ", score)
    print("RMSE score = ", rmse)
    print("MAE score  = ", mae)

def train_models(models, preproc, train_data):
    procdata_X = preproc.fit_transform(train_data.drop(columns="price"))
    for model in models:
        name = ""
        if type(model) is GridSearchCV:
            name = "("+model.estimator.__class__.__name__+")"
        print("Training", model.__class__.__name__, name)
        model.fit(procdata_X, train_data.price)
        #steps = make_pipeline(preproc, model)
        #steps.fit(train_data.drop(columns="price"), train_data["price"])

def train_eval(models, preproc, train_data, test_data, title="", draw=False):
    print("-"*10+title+"-"*10)
    train_models(models, preproc, train_data)
    for i in models:
        eval_model(i, preproc, test_data.drop(columns="price"), test_data["price"])
    print("-"*(20+len(title)))
    if draw:
        draw_cv_results(title+" CV", models)
        
def equal_split(seed, data, models, preproc, ext=False, **args):
    train, test = dp.train_test_equal_split(data, random_state=seed)
    if ext:
        train.drop(columns=["city"], inplace=True)
        test.drop(columns=["city"], inplace=True)
    train_X = preproc.fit_transform(train.drop(columns="price"))
    test_X = preproc.transform(test.drop(columns="price"))
    train_eval(models, preproc, train, test, **args)
    
def diffcities_split(seed, data, models, preproc, ext=False, **args):
    train, test = dp.train_test_diffcities_split(data, random_state=seed)
    if ext:
        train.drop(columns="city", inplace=True)
        test.drop(columns="city", inplace=True)
    train_eval(models, preproc, train, test, **args)
    """fig, ax = plt.subplots()
    ax.set_title(args["title"])
    ax.scatter(test.sqft_living, test.price)
    ax.scatter(test.sqft_living, models[2].predict(preproc.transform(test.drop(columns="price"))))"""
    
if __name__ == "__main__":
    data = dp.Dataset("../data/data.csv")
    
    seed = randint(0, 100_000)
    seed = 19751
    print("Usato il seed:", seed)
    
    preproc = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
        (StandardScaler(), make_column_selector(dtype_include="number"))
        #("passthrough", make_column_selector(dtype_include="number"))
    )
    cvsplits = 5
    
    # Neural network
    mlp = MLPRegressor(hidden_layer_sizes=(10,), max_iter=500, random_state=0, solver="lbfgs")
    grid = {"alpha": [0.0001, 0.001, 0.01, 0.1]}
    mlpcv = GridSearchCV(mlp, grid, cv=cvsplits)
    # Lasso SGD
    sgdlasso = SGDRegressor(penalty="l1", tol=None, alpha=0.001, random_state=0)
    grid = [{"learning_rate": ["invscaling", "adaptive"], "eta0": [0.03, 0.01, 0.001]}]
    sgdlassocv = GridSearchCV(sgdlasso, grid, cv=cvsplits)
    # Ridge SGD
    sgdridge = SGDRegressor(penalty="l2", tol=None, alpha=0.001, random_state=0)
    grid = [{"learning_rate": ["invscaling", "adaptive"], "eta0": [0.03, 0.01, 0.001],
             "alpha": [0.001, 0.01]}]
    sgdridgecv = GridSearchCV(sgdridge, grid, cv=cvsplits)
    # Decision Tree
    tree = DecisionTreeRegressor(random_state=0)
    grid = [{"max_depth": [3, 5, 10], "min_samples_leaf": range(1,10)}]
    treecv = GridSearchCV(tree, grid, cv=cvsplits)
    
    models = [DummyRegressor(), Lasso(), tree]
    # Splitting del dataset normale, dataset originale
    #equal_split(seed, data.original_data, models, preproc, title="ORIGINAL DATA")
    # Splitting del dataset normale, dataset esteso
    #equal_split(seed, data.extended_data, models, preproc, ext=True, title="EXTENDED DATA")
    # Splitting del dataset con splitting tra le città, dataset originale
    diffcities_split(seed, data.original_data, models, preproc, 
                     title="ORIGINAL DATA / DIFFERENT CITIES SPLITTING")
    # Splitting del dataset con splitting tra le città, dataset esteso
    diffcities_split(seed, data.extended_data, models, preproc, ext=True,
                     title="EXTENDED DATA / DIFFERENT CITIES SPLITTING")
    
    #plt.scatter(tr_ordata.sqft_living, tr_ordata.price)
    #plt.scatter(tr_ordata.sqft_living, model.predict(tr_ordata.drop(columns="price")))