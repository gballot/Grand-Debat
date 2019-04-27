import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from src.minisom import MiniSom
import pickle

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
    with open('som.p', 'wb') as outfile:
        pickle.dump(som, outfile)

def open_som():
    with open('som.p', 'rb') as infile:
        return(pickle.load(infile))

def som_training(xsize, ysize, nb_features, sigma, learning_rate, X_train, num_iteration, show_bool):
    # Initialization and training
    som = MiniSom(xsize, ysize, nb_features, sigma=sigma, learning_rate=learning_rate, neighborhood_function='gaussian')

    som.pca_weights_init(X_train)
    print("Training...")
    som.train_batch(X_train, num_iteration, verbose=True)  # random training
    print("\n...ready!")

    plt.figure(figsize=(xsize, ysize))
    # Plotting the response for each pattern in the iris dataset
    plt.pcolor(som.distance_map().T, cmap='bone_r')  # plotting the distance map as background
    #plt.colorbar()

    # use different colors and markers for each label
    markers = ['o', 's', 'D']
    colors = ['C0', 'C1', 'C2']
    for cnt, xx in enumerate(X_train):
        w = som.winner(xx)  # getting the winner
        # palce a marker on the winning position for the sample xx
        plt.plot(w[0]+.5, w[1]+.5, markers[y_train[cnt]], markerfacecolor='None', markeredgecolor=colors[y_train[cnt]], markersize=12, markeredgewidth=2)
    plt.axis([0, xsize, 0, ysize])
    plt.title("Map post-training")
    if(show_bool):
        plt.show()
    #saving som model trained in a file
    save_som(som)

def activation_frequencies(map_size, X_train):
    #open trained som
    som = open_som()
    plt.figure(figsize=(map_size+1, map_size))
    frequencies = np.zeros((map_size, map_size))
    for position, values in som.win_map(X_train).items():
        frequencies[position[0], position[1]] = len(values)
    plt.pcolor(frequencies, cmap='Blues')
    plt.colorbar()
    plt.title("Activation frequence for each neuron")
    plt.show()

def show_cluster(X_train, y_train, map_size):
    #open trained som
    som = open_som()
    labels_map = som.labels_map(X_train, y_train)
    label_names = np.unique(y_train)

    plt.figure(figsize=(map_size, map_size))
    the_grid = GridSpec(map_size, map_size)
    for position in labels_map.keys():
        label_fracs = [labels_map[position][l] for l in label_names]
        plt.subplot(the_grid[6-position[1], position[0]], aspect=1)
        patches, texts = plt.pie(label_fracs)
    plt.legend(patches, label_names, bbox_to_anchor=(0, 3), ncol=3)
    plt.show()

som_training(7, 7, 4, 2.5, 0.5, X_train, 6000, 0)
#activation_frequencies(7, X_train)
show_cluster(X_train, y_train, 7)
