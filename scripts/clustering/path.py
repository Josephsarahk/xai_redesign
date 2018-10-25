from sklearn.cluster import KMeans
from clustering import unpickle
from scipy.spatial.distance import euclidean
from collections import defaultdict
import numpy

def get_cluster_centers(conv_layer_filenames,fc_layer_filenames,n_clusters_conv,n_clusters_fc):
    km = KMeans(n_clusters = n_clusters_conv)
    centers = []
    for name in conv_layer_filenames:
        X = unpickle(name)
        X = list(map(numpy.ndarray.flatten,X))
        km.fit(X)
        centers.append(km.cluster_centers_)
    km.set_params(n_clusters = n_clusters_fc)
    for name in fc_layer_filenames:
        X = unpickle(name)
        X = list(map(numpy.ndarray.flatten,X))
        km.fit(X)
        centers.append(km.cluster_centers_)
    return centers
        
def pathway(activations_list,centers_list):
    path = []
    for i in range(len(centers_list)):
        
        cluster = -1
        dist = numpy.zeros(centers_list[i].shape[0])
        for j in range(centers_list[i].shape[0]):
            dist[j] = euclidean(activations_list[i],centers_list[i][j])
        cluster = numpy.argmin(dist)
        path.append(cluster)
    return path

#path -> incorrect class -> number
def path_distributions(path_list, label_list, output_file):
    paths = {}
    wrong_paths = {}
    f = open(output_file, 'w')
    correct = list(filter(lambda x: x[0] == x[2],path_list))
    incorrect = list(filter(lambda x: x[0] != x[2],path_list))
    path_list = []
    path_list.append(correct)
    path_list.append(incorrect)
    f.write('Correctly labeled:')
    for j in range(len(path_list)):
        f.write('\n\n\n')
        if j == 1:
            f.write('Incorrectly labeled:\n')
        for i in range(len(label_list)):
            if j == 0:
                paths[i] = defaultdict(int)
            else:
                paths[i] = {}
        for i in range(len(path_list[j])):
            label = path_list[j][i][0]
            path = path_list[j][i][1]
            if j == 0:
                paths[label][tuple(path)] = paths[label][tuple(path)] + 1
            else:
                if path_list[j][i][2] in paths[label].keys():
                    paths[label][path_list[j][i][2]][tuple(path)] = paths[label][path_list[j][i][2]][tuple(path)] + 1
                else:
                    paths[label][path_list[j][i][2]] = defaultdict(int)
                    paths[label][path_list[j][i][2]][tuple(path)] = paths[label][path_list[j][i][2]][tuple(path)] + 1
        for label in paths:
            f.write(' ********\n\n' + label_list[label] + ':\n\n ********\n\n')
            count = 0
            if j == 0:
                sorted_list = sorted(paths[label].items(),key = lambda t: (t[1],t[0]))
                sorted_list.reverse()
                for path in paths[label]:
                    count = count + paths[label][tuple(path)]
                for path, frequency in sorted_list:
                #f.write(path, ': ', paths[label][tuple(path)], ' - ', paths[label][tuple(path)]/count, '%')
                    f.write('%s: %d - %.5f%%\n' % (path, paths[label][tuple(path)], (paths[label][tuple(path)]/count)*100))
                f.write('Total examples: ' + str(count) + '\n\n')
            else:
                for wrong_label in paths[label].keys():
                    f.write('Misclassified as %s:\n\n -------------\n\n' % label_list[wrong_label])
                    for path in paths[label][wrong_label].keys():
                        f.write("%s - %d\n" % (path, paths[label][wrong_label][path]))
                    f.write('\n\n -------------\n\n')
    f.close()
    

def strip_list(list):
    return (list[0],list[1:len(list)-1],list[len(list)-1])
