# -*- coding: utf-8 -*-
import random
import datasparql
import os.path as path
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore
from sklearn import model_selection
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline

class Dataset:
    def __init__(self, datapath: str):
        self.original_data = pd.read_csv(datapath)
        print("[+] Dataset base caricato [+]")
        expath = path.normpath(datapath)
        p, f = path.split(expath)
        expath = path.join(p, "ext_"+f)
        if path.exists(expath):
            print("[+] Dataset esteso trovato [+]")
            self.extended_data = pd.read_csv(expath)
        else:
            print("[-] Dataset esteso non trovato [-]")
            self.extended_data = self.__expandData(expath)
        print("[+] Dataset esteso caricato [+]")
    def _expandData(self, expath: str):
        print("[-] Dataset esteso non trovato [-]")
        print("[!] Eseguo query a DBPedia per estendere il dataset base [!]")
        exdata = self.original_data.copy()
        exdata["density"] = 0.
        exdata["lat"] = 0.
        exdata["long"] = 0.
        x = datasparql.citiesDensity(exdata["city"].unique())
        y = datasparql.citiesCoords(exdata["city"].unique())
        print("[+] Feature \"density\" estratta per {}/{} città da DBPedia [+]".format(
            len(x), len(exdata["city"].unique())))
        print("[+] Features \"lat\" e \"long\" estratte per {}/{} città da DBPedia [+]".format(
            len(y), len(exdata["city"].unique())))
        for i, j in exdata.iterrows():
            city = exdata.loc[i, "city"]
            if city in x:
                exdata.loc[i, "density"] = x[city]
            else:
                exdata.loc[i, "density"] = np.fromiter(x.values(), dtype=float).mean()
            if city in y:
                exdata.loc[i, "lat"] = y[city][0]
                exdata.loc[i, "long"] = y[city][1]
        exdata.to_csv(expath, index=False)
        return exdata
    def dropFeatures(self, features: list):
        self.original_data.drop(labels=features, axis=1, inplace=True)
        self.extended_data.drop(labels=features, axis=1, inplace=True)
    def callFunc(self, f):
        f(self.original_data)
        f(self.extended_data)
            
def _set_seeds(seed):
    """
    Per settare i seed prima di ogni operazione randomica, cosi da ottenere
    risultati riproducibili
    """
    random.seed(seed)
    np.random.seed(seed)

def _data_preprocess(data, statezip=False):
    data.drop(index=data[data.price == 0].index, inplace=True)
    data.drop(columns=["date", "street", "country", "waterfront"], inplace=True)
    data.yr_renovated = data[["yr_built", "yr_renovated"]].max(axis=1)
    if not statezip:
        data.statezip = data.statezip.map(lambda x: x.replace("WA ", "")).astype(int)

    #return data[(abs(zscore(data.select_dtypes("number"))) < 3).all(axis=1)]
    return data[(abs(zscore(data.loc[:, :"yr_renovated"])) < 2.5).all(axis=1)]
    
def train_test_equal_split(data, tr_size=0.8, ext=False, random_state=None):
    d = data.copy()
    d = _data_preprocess(d, True)
    
    train, test = model_selection.train_test_split(d, train_size=tr_size, random_state=random_state)
    if ext:
        return train.drop(columns="city"), test.drop(columns="city")
    else:
        return train, test

def train_test_diffcities_split(data, tr_size=0.8, ext=False, random_state=None, verbose=False):
    d = data.copy()
    d = _data_preprocess(d)
    
    if type(random_state) == int:
        _set_seeds(random_state)
    counts = len(d)
    while counts > len(d)*(tr_size+0.05):
        counts = 0
        cities = dict(d.city.value_counts())
        tr_cit = set()
        while counts <= len(d)*tr_size:
            c = random.choice(list(cities.keys()))
            counts += cities.pop(c)
            tr_cit.add(c)
        te_cit = cities
    
    if verbose:
        print("{} città nel set di training, {} città nel set di test".format(len(tr_cit), len(te_cit)))
    train_data = d[d.city.map(lambda x: x in tr_cit)].copy()
    test_data = d[d.city.map(lambda x: x in te_cit)].copy()
    """intersec = train_data.merge(test_data)
    ao = [x for _, x in intersec.groupby(intersec["city"])]
    for cit in ao:
        i = 0
        for _, row in cit.iterrows():
            if i < (2*len(cit))/3:
                test_data.drop(index=test_data[(test_data == row).all(axis=1)].index, inplace=True)
            else:
                train_data.drop(index=train_data[(train_data == row).all(axis=1)].index, inplace=True)
            i += 1"""
    if ext:
        return train_data.drop(columns="city"), test_data.drop(columns="city")
    else:
        return train_data, test_data

def __print_extdata(data):
    fig, ax = plt.subplots(figsize=(9,7))
    lat = list()
    long = list()
    price = list()
    density = list()
    for c, x in data.groupby(data["city"]):
        lat.append(x.lat.mean())
        long.append(x.long.mean())
        price.append(x.price.mean())
        density.append(x.density.mean()/5)
    ax.set_title("Prezzi medi per città")
    ax.set_xlabel("Longitudine")
    ax.set_ylabel("Latitudine")
    points = ax.scatter(long, lat, s=density, c=price)
    fig.colorbar(points)

if __name__ == "__main__":
    data = Dataset("../data/data.csv")
    #print_extdata(_data_preprocess(data.extended_data))
    d = data.original_data
    # Test per scelta delle features attraverso cross validation
    preproc = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), make_column_selector(dtype_exclude="number")),
        (StandardScaler(), make_column_selector(dtype_include="number"))
    )
    model = make_pipeline(preproc, Ridge())
    d = _data_preprocess(d)
    max_val = float("{:.2f}".format(d.corrwith(d.price).drop("price").max()))
    x = np.arange(0, max_val, 0.05)
    y = list()
    for i in x:
        datadrop = abs(d.corrwith(d.price)) < i
        d1 = d.drop(datadrop[datadrop == True].keys().tolist(), axis=1)
        cv = model_selection.cross_validate(model, d1.drop(columns="price"), d1["price"],
                                   scoring=["neg_mean_absolute_error", "r2"], cv=5)
        y.append(-cv["test_neg_mean_absolute_error"].mean())
    plt.suptitle("Feature selection attraverso correlazione delle features")
    plt.plot(x, y, marker="o")