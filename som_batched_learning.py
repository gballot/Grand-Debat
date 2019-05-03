import math
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib3
from sklearn.externals import joblib
import random
import matplotlib
from sompy.sompy import SOMFactory
from sompy.visualization.bmuhits import BmuHitsView
from sompy.visualization.plot_tools import plot_hex_map
from sompy.visualization.hitmap import HitMapView
from sompy.visualization.mapview import View2D
from X import get_X, get_labels, get_full_X, get_auth_id
import logging
import pickle

#Path to savec trained batched SOM
path = './data/batched_SOM_models/'

#Datasets
#from sklearn import datasets
#iris = datasets.load_iris()
#X = iris.data
#y = iris.target

X = get_X()
# Data normalization
#X = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,X)

#Training and test set
#X_train = X[::2]
#y_train = y[::2]
#X_test = X[1::2]
#y_test = y[1::2]
names=get_labels()


# Train nb_models of models of a som map with random size between map_min_size and map_max_size 
# Save the trained models into data/SOM_models 
# Plot the topographic and quantization error
def training_batched_som(map_min_size, map_max_size, nb_models, X_train):
    for i in range(nb_models): 
        sm = SOMFactory().build(X_train, mapsize=[random.choice(list(range(map_min_size, map_max_size))), random.choice(list(range(map_min_size, map_max_size)))], normalization = 'var', initialization='random', component_names=names, lattice="hexa") 
        sm.train(n_job=1, verbose=False, train_rough_len=30, train_finetune_len=100) 
        joblib.dump(sm, path+"batched_model_{}.joblib".format(i))
        print("end of training model n°" + str(i))

    # Study the models trained and plot the errors obtained in order to select the best one
    models_pool = glob.glob(path+"batched_model*")
    errors=[]
    for model_filepath in models_pool:
        sm = joblib.load(model_filepath)
        topographic_error = sm.calculate_topographic_error()
        quantization_error = sm.calculate_quantization_error()
        errors.append((topographic_error, quantization_error)) 
    e_top, e_q = zip(*errors)

    plt.scatter(e_top, e_q)
    plt.xlabel("Topographic error")
    plt.ylabel("Quantization error")
    plt.show()

# Return the best model (best topographic error then best quantization error) into the trained models data/SOM_models
def find_best_model(nb_models):
    models_pool = glob.glob(path+"batched_model*")
    sm = joblib.load(models_pool[0])
    min_topo = sm.calculate_topographic_error()
    min_quanti = sm.calculate_quantization_error()
    best_model = 0
    for i in range(1,nb_models-1):
        selected_model = i
        sm = joblib.load(models_pool[selected_model])
        topographic_error = sm.calculate_topographic_error()
        quantization_error = sm.calculate_quantization_error()
        if (topographic_error < min_topo):
            min_topo = topographic_error
            min_quanti = quantization_error
            best_model = i
        if (topographic_error ==  min_topo):
            if(quantization_error < min_quanti):
                min_quanti = quantization_error
                best_model = i

        #print ("Topographic error = %s,  Quantization error = %s" % (topographic_error, quantization_error))

    sm = joblib.load(models_pool[best_model])
    print("best model is model n°" + str(best_model))
    print("Topographic error: " + str(min_topo))
    print("Quantization error: " + str(min_quanti))
    return(best_model)

def get_best_model(nb_models):
    models_pool = glob.glob(path+"batched_model*")
    return(joblib.load(models_pool[find_best_model(nb_models)]))

def find_clusters(nb_clusters, nb_models):
    sm = get_best_model(nb_models)
    print("som_map_clustered: ",sm.cluster(nb_clusters))
    hits  = HitMapView(12, 12,"Clustering",text_size=10, cmap=plt.cm.jet)
    a=hits.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="Pastel1")
    plt.show()

def get_clusters(nb_models, nb_clusters, X_projected):
    sm = get_best_model(nb_models)
    map_clustered = sm.cluster(nb_clusters)
    projected_data = sm.project_data(X_projected)
    clusters = []
    cluster_0 = [];
    cluster_1 = [];
    cluster_2 = [];
    cluster_3 = [];
    cluster_4 = [];
    cluster_5 = [];
    cluster_6 = [];
    cluster_7 = [];
    cluster_8 = [];
    cluster_9 = [];
    auth_id = get_auth_id()
    for i in range(projected_data.shape[0]):
       clust = map_clustered[projected_data[i]]
       if (clust == 0):
           cluster_0.append(auth_id[i])
       if (clust == 1):
           cluster_1.append(auth_id[i])
       if (clust == 2):
           cluster_2.append(auth_id[i])
       if (clust == 3):
           cluster_3.append(auth_id[i])
       if (clust == 4):
           cluster_4.append(auth_id[i])
       if (clust == 5):
           cluster_5.append(auth_id[i])
       if (clust == 6):
           cluster_6.append(auth_id[i])
       if (clust == 7):
           cluster_7.append(auth_id[i])
       if (clust == 8):
           cluster_8.append(auth_id[i])
       if (clust == 9):
           cluster_9.append(auth_id[i])

    clusters.append(cluster_0)   
    clusters.append(cluster_1)   
    clusters.append(cluster_2)   
    clusters.append(cluster_3)   
    clusters.append(cluster_4)   
    clusters.append(cluster_5)   
    clusters.append(cluster_6)   
    clusters.append(cluster_7)   
    clusters.append(cluster_8)   
    clusters.append(cluster_9)
    return(clusters)

def prototype_visualization(nb_models):
    sm = get_best_model(nb_models)
    view2D  = View2D(4,4,"", text_size=7)
    view2D.show(sm, col_sz=5, which_dim="all", denormalize=True)
    plt.show()

def real_visualization(nb_models, X_train):
    sm = get_best_model(nb_models)
    df = get_full_X()
    df["bmus"] = sm.project_data(X_train)
    df = np.append(df, sm.project_data(X_train), axis=1)
    empirical_codebook=df.groupby("bmus").mean().values
    matplotlib.rcParams.update({'font.size': 10})
    plot_hex_map(empirical_codebook.reshape(sm.codebook.mapsize + [empirical_codebook.shape[-1]]),
             titles=df.columns[:-1], shape=[4, 5], colormap=None)
    plt.show()

def hit_map(nb_models):
    sm = get_best_model(nb_models)
    vhts  = BmuHitsView(12,12,"Hits Map",text_size=7)
    vhts.show(sm, anotate=True, onlyzeros=False, labelsize=7, cmap="autumn", logaritmic=False)
    plt.show()


nb_models = 1000

#training_batched_som(map_min_size=10, map_max_size=50, nb_models=nb_models, X_train=X)
#find_clusters(nb_clusters=10, nb_models=nb_models)
#prototype_visualization(nb_models=nb_models)
#real_visualization(nb_models=nb_models, X_train=X)
#hit_map(nb_models=nb_models)
get_clusters(nb_clusters=10, nb_models=10, X_projected=X)
