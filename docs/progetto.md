# Indice
1. **Progetto**
	1. Requisiti
	2. Files
2. **Analisi dei dati, processing e conoscenza dal web semantico**
	1. Estensione del dataset
	2. Classe Dataset
	3. Trasformazione delle features
	4. Metodi per lo splitting del dataset
3. **Modelli di regressione**
	1. Apprendimento supervisionato
	2. Confronto dei risultati tra dataset
	3. Clustering

---

# 1. Progetto
Lo scopo di questo progetto è costruire dei modelli di regressione per la
predizione di prezzi di alcune case, partendo da caratteristiche della casa e
del luogo dove si trova. Inoltre vengono confrontate le performace di modelli
allenati sul dataset originale e su un dataset esteso, al quale sono state
aggiunte alcune features attraverso query a **[DBpedia](http://dbpedia.org)**,
quindi si vedrà se è possibile migliorare le predizioni dei prezzi attraverso
l'integrazione di conoscenza proveniente dal web semantico.

## 1.1 Requisiti
Il progetto è stato scritto interamente in **Python 3** e sono state usate le
seguenti librerie:

- numpy
- scikit-learn
- pandas
- sparqlwrapper

## 1.2 Files
La repository è organizzata nel seguente modo:

- **data**: in questa cartella sono contenuti i files .csv contenenti il
dataset (originale ed esteso)
- **src**: in questa cartella è presente il codice sorgente del progetto, in
particolare:
	- **`datasparql.py`**: contiene il codice per effettuare le query a DBpedia
	- **`data_preprocessing.py`**: contiene il codice relativo al processing
	dei dati e per la gestione del dataset
	- **`models.py`**: contiene il codice relativo alla costruzione dei vari
	modelli di apprendimento e gestisce il training ed il testing
- **docs**: contiene la documentazione del progetto
- **img**: contiene i grafici e le varie immagini

# 2. Analisi dei dati, processing e conoscenza dal web semantico
Nel dataset originale (presente in _data/data.csv_), sono presenti i seguenti
attributi:

- **date** (la data di rilevazione)
- **price** (il prezzo della casa)
- **bedrooms** (il numero di stanze da letto)
- **bathrooms** (il numero di bagni)
- **sqft_living** (dimensione della casa)
- **sqft_lot** (dimensione del terreno)
- **floors** (numero di piani)
- **waterfront** (se la casa è sul mare)
- **view** (voto della vista)
- **condition** (condizione della casa)
- **sqft_above** (dimensione della casa escluse stanze sotterranee)
- **sqft_basement** (dimensione di un eventuale piano sotterraneo)
- **yr_built** (anno di costruzione)
- **yr_renovated** (anno di ristrutturazione)
- **street** (via)
- **city** (città)
- **statezip** (codice zip)
- **country** (nazione)

## 2.1 Estensione del dataset
nel file `data/ext_data.csv` è presente il dataset che è stato esteso
attraverso delle query a DBpedia. Le query vengono effettuate nel modulo
`datasparql.py`, in particolare la funzione:
``` python
def citiesDensity(cities: list)
```
permette di estrarre la densità abitativa di una lista di città. Invece la
funzione:
``` python
def citiesCoords(cities: list)
```
permette di estrarre latitudine e longitudine delle città, questo permetterà di
sostituire la feature **city** con le features **lat** e **long** e quindi
rappresentare la città non come una feature categorica ma come un valore
numerico. Un altro vantaggio è che la predizione per una casa in una città che 
non è presente nel set di training sarà più precisa grazie all'uso di
latitudine e longitudine.

Quindi le features aggiunte nel dataset esteso sono:

- **lat**
- **long**
- **density**

La posizione della casa effettivamente influisce sul prezzo, mentre la densità
sembra influire meno, come si può vedere dal seguente grafico:

![città](imgs/cities.png)
_Case raggruppate per città, il colore rappresenta il prezzo medio delle case,
la grandezza del punto mostra la densità abitativa della città._
