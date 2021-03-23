# Clustering enquesta d'associats

Clustering mixte numèric categòric ambn els resultats de les dades de l'enquesta d'associats.


## Instal·lació (millor en virtualenv)

(Passos 1 i 2 només si s'utilitza virtualenv)

### 1. Crear virtualenv al directori ~/venv (crear-lo si no existeix. També pot ser una altra ubicació):
virtualenv --python=python3.7 ~/venv/cv

### 2. Activar virtualenv:
source ~/venv/cv/bin/activate

### 3. Instal·lar els paqueets necessaris
pip install kmodes
pip install pandas
pip install numpy
pip install matplotlib

## Execució:
python clustering_enquesta_associats.py

## Anàlisi dels resultats:

Carregar el csv resultant en Google DataStudio (https://datastudio.google.com/) i crear 'radar charts'

