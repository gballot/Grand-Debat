import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.minisom import MiniSom
import pickle
from sklearn.externals import joblib
import random
import glob

#Path to the savec unbatched SOM
path = './data/unbatched_SOM_models/'

#Datasets
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Data normalization
X = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,X)

#Training and test set
X_train = X[::2]
y_train = y[::2]
X_test = X[1::2]
y_test = y[1::2]

def save_som(som):
    with open(path+'som.p', 'wb') as outfile:
        pickle.dump(som, outfile)

def open_som():
    with open(path+'som.p', 'rb') as infile:
        return(pickle.load(infile))

#Train only one map model and plot it
def som_solo_training(xsize, ysize, nb_features, sigma, learning_rate, X_train, y_train, num_iteration, show_bool):
    # Initialization and training
    som = MiniSom(xsize, ysize, nb_features, sigma=sigma, learning_rate=learning_rate, neighborhood_function='gaussian')

    som.pca_weights_init(X_train)
    print("Training...")
    som.train_batch(X_train, num_iteration, verbose=True)  # random training
    print("\r...ready!")

    plt.figure(figsize=(xsize, ysize))
    # Plotting the response for each pattern in the iris dataset
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    #plt.colorbar()

    # use different colors and markers for each label
    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        # place a marker on the winning position for the sample xx
        plt.plot(w[0]+.5, w[1]+.5, markers[y_train[cnt]], markerfacecolor='None', markeredgecolor=colors[y_train[cnt]], markersize=12, markeredgewidth=2)
    plt.axis([0, xsize, 0, ysize])
    plt.title("Map post-training")
    if(show_bool):
        plt.show()
    #saving som model trained in a file
    save_som(som)

#Train nb_models of maps whose size in randomly between map_min_size and map_max_size
def som_training(map_min_size, map_max_size, nb_features, sigma, learning_rate, X_train, num_iteration):
    # Initialization and training
    for i in range(map_min_size,map_max_size):
        som = MiniSom(i, i, nb_features, sigma=sigma, learning_rate=learning_rate, neighborhood_function='gaussian')
        som.pca_weights_init(X_train)
        som.train_batch(X_train, num_iteration, verbose=True)  # random training
        print("End of training model n°" + str(i)+ " of size "+str(i)+'*'+str(i))
        #saving som model trained in a file
        joblib.dump(som, path+"unbatched_model_{}.joblib".format(i))


# Return the best model into the trained models data/unbatched_SOM_models
def find_best_model(nb_models, X_train):
    models_pool = glob.glob(path+"unbatched_model*")
    som = joblib.load(models_pool[0])
    min_quanti = som.quantization_error(X_train)
    best_model = 0
    for i in range(1,nb_models-1):
        selected_model = i
        som = joblib.load(models_pool[selected_model])
        quanti_error = som.quantization_error(X_train)
        if (quanti_error < min_quanti):
            min_quanti = quanti_error
            best_model = i

#       if (quanti_error ==  min_quanti):
#                actual_best_som = joblib.load(models_pool[best_model])
#                som_size = [som.x, som.y]
#                actual_best_som_size = [actual_best_som.x, actual_best_som.y]
#                if(som_size < actual_best_som_size):
#                    min_quanti = quanti_error
#                    best_model = i

    sm = joblib.load(models_pool[best_model])
    print("Best model is model n°" + str(best_model))
    print("Quantization error: " + str(min_quanti))
    return(sm)


def plot_som(nb_models, X_train, y_train):
    som = find_best_model(nb_models, X_train)    
    # use different colors and markers for each label
    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        # place a marker on the winning position for the sample xx
        plt.plot(w[0]+.5, w[1]+.5, markers[y_train[cnt]], markerfacecolor='None', markeredgecolor=colors[y_train[cnt]], markersize=12, markeredgewidth=2)
    #plt.axis([0, xsize, 0, ysize])
    plt.title("Map post-training")
    plt.show()


# Plot the activation frequencies of a SOM
def activation_frequencies(map_size, X, som_type, nb_models, show_bool):
    #import the best model found after som_training
    if(som_type == 'best'): 
       som = find_best_model(nb_models, X_train)    
    #import a som after a solo training
    if(som_type == 'solo'):
        som = open_som()
    
    plt.figure(figsize=(map_size+1, map_size))
    frequencies = np.zeros((map_size, map_size))
    for position, values in som.win_map(X).items():
        frequencies[position[0], position[1]] = len(values)
    plt.pcolor(frequencies, cmap='Blues')
    plt.colorbar()
    plt.title("Activation frequence for each neuron")
    if(show_bool):
        plt.show()

#Show clusters of a SOM
def show_cluster(map_size, X, y, som_type, nb_models, show_bool):
     #import the best model found after som_training
    if(som_type == 'best'): 
        som = find_best_model(nb_models, X_train)    
    #import a som after a solo training
    if(som_type == 'solo'):
        som = open_som()

    labels_map = som.labels_map(X, y)
    label_names = np.unique(y)

    plt.figure(figsize=(map_size, map_size))
    the_grid = GridSpec(map_size, map_size)
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names]
        plt.subplot(the_grid[5-position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
    plt.legend(patches, label_names, bbox_to_anchor=(0, 3), ncol=3)
    if(show_bool):
        plt.show()

#som_solo_training(4, 4, 4, 0.5, 0.5, X, y, 10000, 0)
#som_training(map_min_size=2, map_max_size=10, nb_features=4, sigma=0.5, learning_rate=0.5, X_train=X, num_iteration=10000)
plot_som(nb_models=8, X_train=X, y_train=y) 
activation_frequencies(map_size=9, X=X, som_type='best', nb_models=8, show_bool=0)
show_cluster(map_size=9, X=X, y=y, som_type='best', nb_models=8, show_bool=1)
