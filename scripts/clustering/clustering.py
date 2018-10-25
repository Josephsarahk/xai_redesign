import pickle
from matplotlib.pyplot import imsave
import numpy as np

"""
Simple unserialization function.
Params:
    :pathname - string representing file location to load from
"""
def unpickle(pathname):
    with open(pathname, 'rb') as file:
        return pickle.load(file)

"""
Prints distribution of samples in each cluster according to label.
Params:
   :clusterer - sklearn clustering object
   :cluster_labels - cluster labels for each point provided as numpy.array
   :classes - string names for each label in the dataset being used provided as list or array
   :prediction_labels - predicted labels for each point provided as list or array
   :true_labels - ground truth label for each point provided as list or array
   :output_name - string representing file to write to. Only plain text will be written to this file;
   include file type
   :is_images - flag that allows for visualization of cluster. If set to True, 
   MUST provide all optional parameters, then the function will provide visualizations
   of the cluster centers in activation space.
   :centers - centers of each cluster, provided as numpy.array
   :dimensions - tuple representing the original shape of the activation matrices
   :centers_output - string representing prefix of filenames to give to each image.
   Files will be named <centers_output>_CLUSTER-NO_FILTER-NO.png.
"""
def print_distribution(clusterer, cluster_labels, classes, prediction_labels, true_labels, output_name, is_images = False, centers = None, dimensions = None,centers_output=None):
    f = open(output_name, 'w')
    dict = {}
    misclass=  []
    for i in range(clusterer.get_params()['n_clusters']):
        dict[i] = {}
        misclass.append(0)
        if(is_images == True):
            img = centers[i].reshape(*dimensions) #filters*dimensions
            for filters in range(img.shape[0]):
                imsave(centers_output + "_" + str(i) + "_" + str(filters) + ".png",img[filters],cmap='gray')
        for j in range(len(classes)):
            dict[i][j] = 0
    for k in range(cluster_labels.size):
        dict[cluster_labels[k]][prediction_labels[k]] = dict[cluster_labels[k]][prediction_labels[k]] + 1
        if(prediction_labels[k] != true_labels[k]):
            misclass[cluster_labels[k]] = misclass[cluster_labels[k]] + 1
    for k, v in dict.items():
        counter = 0
        f.write("Cluster %d:\n" % k)
        for name, num in v.items():
            f.write("%d %s\n" % (num, classes[name]))
            counter=counter+num
        f.write("Number misclassified: %d/%d\n" % (misclass[k],counter))
    f.close()

#X - array of channel activations, vectorized
#cluster_labels - array of cluster labels

def compare_filter_dist(center1, center2, num_filters):
    split1 = np.split(center1,num_filters)
    split2 = np.split(center2,num_filters)
    dists = np.zeros((num_filters))
    for i in range(len(split1)):
        dists[i] = np.linalg.norm(split1[i]-split2[i])
    return dists
    
def compare_all_centroids(centers,num_filters,filename):
    f = open(filename,'w')
    counts = np.zeros((num_filters),dtype='int')
    second_counts = np.zeros((num_filters),dtype='int')
    for i in range(centers.shape[0]):
        for j in range(i+1,centers.shape[0]):
            filter_dists = compare_filter_dist(centers[i],centers[j],num_filters)
            most_sens = np.argmax(filter_dists)
            second = np.argmax(filter_dists[filter_dists != filter_dists[most_sens]])
            second = second if second < most_sens else second + 1
            f.write('Cluster ' + str(i) + ' and Cluster ' + str(j) + ': ' + str(filter_dists) + '\n')
            f.write('Most sensitive filter: ' + str(most_sens) + '\n')
            f.write('Second most sensitive: ' + str(second) + '\n')
            counts[most_sens] = counts[most_sens] + 1
            second_counts[second] = second_counts[second] + 1
    f.write('Final counts:\n')
    for i in range(counts.size):
        f.write(str(i) + ': ' + str(counts[i]) + '\n')
    f.write('Runner-ups:\n')
    for i in range(counts.size):
        f.write(str(i) + ': ' + str(second_counts[i]) + '\n')
    f.close()
        
    
    
#Use sklearn's clustering module to do clustering. The printout tool here is oriented for Nitin's data