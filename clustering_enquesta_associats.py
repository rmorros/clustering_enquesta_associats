import sys
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
import matplotlib.pyplot as plt

file_name = 'respostes_enquesta_associats_postprocessades.csv'

# For the clustering, we will differentiate between numeric variables and categorical variables. 
# A categorical variable (sometimes called a nominal variable) is one that has two or more categories,
# but there is no intrinsic ordering to the categories.
# An ordinal variable is similar to a categorical variable.  The difference between the two is that
# there is a clear ordering of the categories.
# A numerical variable is similar to an ordinal variable, except that the intervals between the values
# of the numerical variable are equally spaced.

# Type of data 'N': numerical, 'C': categorical, 'CM': categorical with multiple options, 'NP': numeric with missing answers
data_type =  ['N', 'C', 'N', 'CM', 'C', 'CN?', 'C','C', 'C', 'NP', 'C', 'CM', 'NP', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'NP', 'N', 'N', 'C', 'C']


df = pd.read_csv(file_name, sep=',')
df_ori = df.copy()  # Keep the original dataframe


# Some fields are personal opinions and not suitable for clustering. Remove them.
df.drop(['X1','X2','X3','ESCOLA'], axis='columns', inplace=True)

# The column 'ACREDITACIO' can have multiple answers.
# keep only the 'best' accreditation if more than one
acreditations_names_dic = {'Universitat privada':1, 'Col·laborador/a':2, 'Ajudant doctor':3, 'Lector/a':4, 'Contractat/da doctor/a':5, 'Agregat/da':6}


# Simplify ACREDITACIO column: keep only the 'best' accreditation if more than one
acred_column = list(df['ACREDITACIO'])

simplified_acred = []
for elem in acred_column:
    codes = [0]
    for name,val in acreditations_names_dic.items():
        if name in elem:
            codes.append(val)
    simplified_acred.append(sorted(codes, reverse=True)[0])  # Keep only the maximum value (most important acreditation)

df.drop('ACREDITACIO', axis='columns', inplace=True)
df['ACREDITACIO'] = simplified_acred


head = ['EDAT', 'SEXE', 'ANTIGUETAT', 'DEPARTAMENT', 'TIPUS', 'ALTRAUNI', 'REDUCCIO', 'DOCTOR', 'ANYTESI', 'RAOTESI', 'ACREDITACIO', 'ANYACREDITACIO', 'FENTTESI', 'RAOFENTTESI', 'DISPOSATTESI', 'CARRERA', 'ASSOCIAT', 'FALSASSOCIAT', 'EXPECTATIVES', 'GESTIO', 'RENOVACIO', 'BONTRACTE', 'FORMACIO', 'SATISFET', 'ASSEMBLEES', 'MOBILITZACIONS']

# Drop ['ALTRAUNI', 'REDUCCIO', 'RENOVACIO', 'ASSEMBLEES', 'MOBILITZACIONS'], which may not be useful for clustering
df.drop(['ALTRAUNI', 'REDUCCIO', 'RENOVACIO', 'ASSEMBLEES', 'MOBILITZACIONS', 'BONTRACTE', 'DEPARTAMENT', 'ANYACREDITACIO', 'ANYTESI', 'GESTIO'] , axis='columns', inplace=True)


# Headers for the columns in the adapted sheet
df_columns = ['EDAT', 'SEXE', 'ANTIGUETAT', 'TIPUS', 'DOCTOR', 'RAOTESI', 'ACREDITACIO', 'FENTTESI', 'RAOFENTTESI', 'DISPOSATTESI', 'CARRERA', 'ASSOCIAT', 'FALSASSOCIAT', 'EXPECTATIVES', 'FORMACIO', 'SATISFET']


# To perform clustering, it is easier to represent the categories in numerical form
# We use label encoding.
# See https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd

# Convert data to category codes (numerical)
for col in df_columns:
    df[col] = df[col].astype('category').cat.codes


#df.to_csv('out_simple_kp.csv')


# In the form, there are conditional questions (they only appear depending of the answer to
# a previous question). For instance, if the user answers Yes to DOCTOR, then he/she can answer
# questions 'ANYTESI', 'RAOTESI', 'ACREDITACIO', 'ANYACREDITACIO'. If the user answers No,
# the extra choices are 'FENTTESI', 'RAOFENTTESI', 'DISPOSATTESI'.
# Because of this, two main grups are created, DOCTOR=Yes, DOCTOR=No

# Separate doctors from non-doctors. The two partitions will be clustered separately
df_sidoctor = df.loc[df['DOCTOR'] == 1].copy()
df_nodoctor = df.loc[df['DOCTOR'] == 0].copy()

# Drop fields that do not apply in each case
df_nodoctor.drop(['DOCTOR', 'RAOTESI', 'ACREDITACIO'      ],    axis='columns', inplace=True)
#df_nodoctor.drop(acreditations_short_names,                                axis='columns', inplace=True)
df_sidoctor.drop(['DOCTOR', 'FENTTESI', 'RAOFENTTESI', 'DISPOSATTESI'],    axis='columns', inplace=True)


#df_sidoctor.to_csv('out_sidoctor_simple_kp.csv')
#df_nodoctor.to_csv('out_nodoctor_simple_kp.csv')


# New set of headers for the columns after changes
no_doctor_head = ['EDAT', 'SEXE', 'ANTIGUETAT', 'TIPUS', 'FENTTESI', 'RAOFENTTESI', 'DISPOSATTESI', 'CARRERA', 'ASSOCIAT', 'FALSASSOCIAT', 'EXPECTATIVES', 'FORMACIO', 'SATISFET']

# New set of headers for the columns after changes
si_doctor_head = ['EDAT', 'SEXE', 'ANTIGUETAT', 'TIPUS', 'RAOTESI', 'ACREDITACIO', 'CARRERA', 'ASSOCIAT', 'FALSASSOCIAT', 'EXPECTATIVES', 'FORMACIO', 'SATISFET']

# Encoding of type of columns. 0 means the column is numeric, 1 means the column is categorical.
no_doctor_cat = [0,1,0,1, 1,1,1, 1,1,1,1,1,0]
si_doctor_cat = [0,1,0,1,   1,1, 1,1,1,1,1,0]

# Indices of categorical columns
no_doctor_cat_list = [ii for ii, value in enumerate(no_doctor_cat) if value == 1]
si_doctor_cat_list = [ii for ii, value in enumerate(si_doctor_cat) if value == 1]

# Convert data to numpy arrays (required by KPrototypes)
no_doctor_data = df_nodoctor.to_numpy(copy=True)
si_doctor_data = df_sidoctor.to_numpy(copy=True)


#print (df_nodoctor)
#print ('------------------------------------')
#print (df_nodoctor.index.values)
#print ('------------------------------------')
#print (no_doctor_data)
#print ('------------------------------------')

ind_nodoctor = df_nodoctor.index.values
ind_sidoctor = df_sidoctor.index.values


no_doctor_clusters  = []
no_doctor_centroids = []
no_doctor_labels    = []
no_doctor_cost      = []


# https://www.researchgate.net/deref/http%3A%2F%2Fciteseerx.ist.psu.edu%2Fviewdoc%2Fdownload%3Fdoi%3D10.1.1.15.4028%26rep%3Drep1%26type%3Dpdf

# Selecting the number of clusters can be tricky. In this case, we will try several 
# values of K and check visually which one offers the best explanaibility for our case.


# Perform clustering with several values of K for non-doctors
df_nodoctor_clust = []
for ii in range(2,6):
    print ('Non doctors: using {} clusters'.format(ii))
    kp = KPrototypes(n_clusters=ii, init='Cao', verbose=2)
    # km = KModes(n_clusters=ii, init='Huang', n_init=5, verbose=0)
    no_doctor_clusters.append(kp.fit_predict(no_doctor_data, categorical=no_doctor_cat_list))
    no_doctor_centroids.append(kp.cluster_centroids_)
    no_doctor_labels.append(kp.labels_)
    no_doctor_cost.append(kp.cost_)


    df_nodoctor_clust.append(df_nodoctor)
    df_nodoctor_clust[-1]['Cluster'] = no_doctor_clusters[-1]

    # Save the results
    out_name = 'out_nodoctor_k{}_simple_kp.csv'.format(ii)
    df_nodoctor_clust[-1].to_csv(out_name)



si_doctor_clusters  = []
si_doctor_centroids = []
si_doctor_labels    = []
si_doctor_cost      = []

# Perform clustering with several values of K for doctors
df_sidoctor_clust = []
for ii in range(2,6):
    print ('Doctors: using {} clusters'.format(ii))
    kp = KPrototypes(n_clusters=ii, init='Cao', verbose=2)
    si_doctor_clusters.append(kp.fit_predict(si_doctor_data, categorical=si_doctor_cat_list))

    df_sidoctor_clust.append(df_sidoctor)
    df_sidoctor_clust[-1]['Cluster'] = si_doctor_clusters[-1]

    # Save the results
    out_name = 'out_sidoctor_k{}_simple_kp.csv'.format(ii)
    df_sidoctor_clust[-1].to_csv(out_name)
    

# To analyze the results, a good option is to use  Google DataStudio (https://datastudio.google.com/)
# with radar charts
