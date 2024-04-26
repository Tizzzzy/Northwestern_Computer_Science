import random
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import scale 
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns



# from k_nearest_neighbor import KNearestNeighbor
# returns Euclidean distance between vectors a dn b
def euclidean(a,b):
    # 这是写的
    squared_diff = [(x-y) ** 2 for x, y in zip(a, b)]
    sum_squared_diff = sum(squared_diff)

    dist = sum_squared_diff ** 0.5

    # 这是np检查
    dist_np = np.linalg.norm(a - b)

    # print(dist, dist_np)
    
    return(dist)
        
# returns Cosine Similarity between vectors a dn b
def cosim(a,b):
    # 这是写的
    dot_product = sum(p * q for p, q in zip(a, b))
    magnitude_a = sum(p**2 for p in a)**0.5
    magnitude_b = sum(q**2 for q in b)**0.5

    dist = dot_product/(magnitude_a*magnitude_b)
    
    # 这是np检查
    dist_np = np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))
    # print(dist, dist_np)

    return(dist)

def accuarcy(labels):
    count = 0
    total = len(labels)
    for label in labels:
        predict = label[0]
        true = label[1]
        if int(predict) == int(true):
            count += 1
    accuarc = count/total
    return accuarc

def calculate_accuracy(true_labels, predicted_labels):
    assert len(true_labels) == len(predicted_labels), "Lists must be of the same length"
    correct_predictions = sum(t == p for t, p in zip(true_labels, predicted_labels))
    total_predictions = len(true_labels)
    accuracy = correct_predictions / total_predictions
    return accuracy

def plot_confusion_matrix(true_label, predict_label, labels):
    cm = confusion_matrix(true_label, predict_label, labels=labels)

    # Visualize the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def plot_kmeans(clusters):
    fig, axes = plt.subplots(10, 1, figsize=(16, 8))
    axes = axes.flatten()

    for i, cluster in enumerate(clusters):
        # Convert the cluster to a 2D array (10x784) for plotting
        cluster_data = np.vstack(cluster)

        # Plot the data as an image
        axes[i].imshow(cluster_data, cmap='gray')
        axes[i].set_title(f'Cluster {i + 1}')

    # Adjust spacing between subplots for better visualization
    plt.tight_layout()

    # Show the plot
    plt.show()

def choose_k(train, validation, metric):
    best_k = 0
    best_acc = 0
    for k in range(1, 12):
        labels = []
        for obs in range(len(validation)):
            true_label = validation[obs][0]
            data = validation[obs][1]
            data = np.array(data, dtype=np.int32)
            distance = []
            for obs_t in range(len(train)):
                train_data = train[obs_t]
                label = train_data[0]
                value = train_data[1]
                value = np.array(value, dtype=np.int32)
                if(metric == 'euclidean'):
                    calculate_distance = euclidean(value, data)
                    distance.append((calculate_distance, label))
                elif(metric == 'cosim'):
                    calculate_distance = cosim(value,data)
                    distance.append((calculate_distance, label))

            if(metric == 'euclidean'):
                distance = sorted(distance)[:k]
            elif(metric == 'cosim'):
                distance = sorted(distance, reverse=True)[:k]
                
            count_dict = defaultdict(int)
            max_count = 0
            predict = None

            for item in distance:
                count_dict[item[1]] += 1
                if count_dict[item[1]] > max_count:
                    max_count = count_dict[item[1]]
                    predict = item[1]
            labels.append((predict, true_label))

        accurate = accuarcy(labels)
        print("k: ", k, "accuracy: ", accurate, "len: ", len(labels))
        if accurate >= best_acc:
            best_acc = accurate
            best_k = k

    
    return best_k

# returns a list of labels for the query dataset based upon labeled observations in the train dataset.
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def knn(train,query,metric):
    if metric == 'euclidean':
        k = 2
    elif metric == 'cosim':
        k = 3
    labels = []
    true_labels = []
    # k = 11
    for obs in range(len(query)):
        true_label = query[obs][0]
        data = query[obs][1]
        data = np.array(data, dtype=np.int32)
        distance = []
        for obs_t in range(len(train)):
            train_data = train[obs_t]
            label = train_data[0]
            value = train_data[1]
            value = np.array(value, dtype=np.int32)
            if(metric == 'euclidean'):
                calculate_distance = euclidean(value, data)
                distance.append((calculate_distance, label))
            elif(metric == 'cosim'):
                calculate_distance = cosim(value,data)
                distance.append((calculate_distance, label))

        if(metric == 'euclidean'):
            distance = sorted(distance)[:k]
        elif(metric == 'cosim'):
            distance = sorted(distance, reverse=True)[:k]

        count_dict = defaultdict(int)
        max_count = 0
        predict = None

        for item in distance:
            count_dict[item[1]] += 1
            if count_dict[item[1]] > max_count:
                max_count = count_dict[item[1]]
                predict = item[1]
        labels.append(predict)
        true_labels.append(true_label)

    
    return labels, true_labels



# returns a list of labels for the query dataset based upon observations in the train dataset. 
# labels should be ignored in the training set
# metric is a string specifying either "euclidean" or "cosim".  
# All hyper-parameters should be hard-coded in the algorithm.
def kmeans(train,query,metric):
    k = 10
    labels = []
    train_value = []
    train_label = []
    query_value = []
    query_label = []
    labels_query = []
    for label, value in train:
        train_value.append(value)
        train_label.append(label)
    for label, value in query:
        query_value.append(value)
        query_label.append(label)

    # train = np.array(train, dtype=np.int32)
    train_value = np.array(train_value, dtype=np.int32)
    train_label = np.array(train_label, dtype=np.int32)
    query_value_np = np.array(query_value, dtype=np.int32)
    query_label_np = np.array(query_label, dtype=np.int32)

    centroids_label = []
    while len(set(centroids_label)) < k:
        index = np.random.choice(len(train_value), size=k, replace=False)
        centroids = [train_value[i] for i in index]
        centroids_label = [train_label[i] for i in index]
    # print(centroids_label)

    # print('centroids: ', centroids)
    # print('hehrehrehrhehrh: ', centroids_label)

    # print(centroids)
    if metric == 'euclidean':
        rang = 20
    if metric == 'cosim':
        rang = 30
    for i in range(rang):
        clusters = [[] for _ in range(k)]
        for value in train_value:
            if metric == 'euclidean':
                distance = [euclidean(value, c) for c in centroids]
                min_index = np.argmin(distance)
            elif metric == 'cosim':
                distance = [cosim(value, c) for c in centroids]
                min_index = np.argmax(distance)
            clusters[min_index].append(value)
        for j in range(k):
            cluster = clusters[j]
            if len(cluster) > 0:
                centroids[j] = np.nanmean(cluster, axis=0)
            else:
                # Handle empty clusters by re-initializing centroids randomly
                index = np.random.choice(len(train_value), 1)
                centroids[j] = train_value[index]
                centroids_label[j] = train_label[index]

    query_clusters = [[] for _ in range(k)]
    for value in query_value_np:
        distance = [euclidean(value, c) for c in centroids]
        min_index = np.argmin(distance)
        query_clusters[min_index].append((centroids_label[min_index], value))

    count = 0
    correct_num = 0
    total_num = len(query)
    for count in range(k):
        for value in query_clusters[count]:
            predict = value[0]
            # print(value[1])
            for ind, val in enumerate(query_value_np):
                # print(val)
                # print(value[1])
                if np.array_equal(val, value[1]):
                    true = query_label[ind]
                    break
            # true_index = np.where(query_value == value[1])[0]
            # true = query_label[true_index]
            # print(predict, true)
            # print(value[1])
            if predict == int(true):
                correct_num += 1
    return correct_num/total_num
    
    # with open('temp.txt', 'w') as file:
    #     for line in query_clusters:
    #         # line = ' '.join(map(str, line))
    #         file.write(str(line))
    #         file.write('\n')

    # plot_kmeans(query_clusters)
    

def soft_kmeans(train,query,metric,beta=1.0):

    k = 10
    labels = []
    train_value = []
    train_label = []
    query_value = []
    query_label = []
    labels_query = []
    for label, value in train:
        train_value.append(value)
        train_label.append(label)
    for label, value in query:
        query_value.append(value)
        query_label.append(label)

    train_value = np.array(train_value, dtype=np.int32)
    train_label = np.array(train_label, dtype=np.int32)
    query_value_np = np.array(query_value, dtype=np.int32)
    query_label_np = np.array(query_label, dtype=np.int32)

    centroids_label = []
    while len(set(centroids_label)) < k:
        index = np.random.choice(len(train_value), size=k, replace=False)
        centroids = [train_value[i] for i in index]
        centroids_label = [train_label[i] for i in index]
    # print(centroids_label)

    for i in range(20):
        # Initialize an empty array to hold the soft assignments
        soft_assignments = np.zeros((len(train_value), k))
        
        # Calculate the soft assignments for each data point
        for j, value in enumerate(train_value):
            if metric == 'euclidean':
                distances = [euclidean(value, c) for c in centroids]
            elif metric == 'cosim':
                distances = [cosim(value, c) for c in centroids]
            weights = np.exp(-beta * np.array(distances))
            weights /= (weights.sum(axis=0) + 1e-10)
            weights = weights.mean()
            soft_assignments[j] = weights

        # Update the centroids based on the soft assignments
        for j in range(k):
            weights = soft_assignments[:, j][:, None]
            if weights.sum(axis=0) > 0:
                centroids[j] = (weights * train_value).sum(axis=0) / weights.sum(axis=0)
            else:
                # Handle empty clusters by re-initializing centroids randomly
                index = np.random.choice(len(train_value), 1)
                centroids[j] = train_value[index]
                centroids_label[j] = train_label[index]

    query_clusters = [[] for _ in range(k)]
    for value in query_value_np:
        distance = [euclidean(value, c) for c in centroids]
        min_index = np.argmin(distance)
        query_clusters[min_index].append((centroids_label[min_index], value))

    count = 0
    correct_num = 0
    total_num = len(query)
    for count in range(k):
        for value in query_clusters[count]:
            predict = value[0]
            # print(value[1])
            for ind, val in enumerate(query_value_np):
                # print(val)
                # print(value[1])
                if np.array_equal(val, value[1]):
                    true = query_label[ind]
                    break

            # print(predict, true)
            # print(value[1])
            if int(predict[0]) == int(true):
                correct_num += 1
    return correct_num/total_num


def read_data(file_name):
    
    data_set = []
    with open(file_name,'rt') as f:
        for line in f:
            line = line.replace('\n','')
            tokens = line.split(',')
            label = tokens[0]
            attribs = []
            for i in range(784):
                attribs.append(tokens[i+1])
            data_set.append([label,attribs])
    return(data_set)
        
def show(file_name,mode):
    
    data_set = read_data(file_name)
    for obs in range(len(data_set)):
        for idx in range(784):
            if mode == 'pixels':
                if data_set[obs][1][idx] == '0':
                    print(' ',end='')
                else:
                    print('*',end='')
            else:
                print('%4s ' % data_set[obs][1][idx],end='')
            if (idx % 28) == 27:
                print(' ')
        print('LABEL: %s' % data_set[obs][0],end='')
        print(' ')
            
def main():
    # a = np.array([2,1,2,3,2,9])
    # b = np.array([3,4,2,4,5,5])
    # eucli = euclidean(a, b)
    # cos = cosim(a, b)
    # print(eucli, cos)
    # show('valid.csv','pixels')
    train = read_data('train.csv')
    valid = read_data('valid.csv')
    query = read_data('test.csv')   #test data有200个
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    # Choose best k:
    '''
    k = choose_k(train, valid, 'cosim')
    print('k: ', k)
    '''

    # KNN euclidean
    predict, true = knn(train, query, 'euclidean')
    accuarc = calculate_accuracy(true, predict)
    print('KNN euclidean: ', accuarc)

    # KNN cosim
    predict, true = knn(train, query, 'cosim')
    accuarc = calculate_accuracy(true, predict)
    print('KNN cosim: ',accuarc)

    # plot KNN confusion matrix
    # plot_confusion_matrix(true, predict, labels)
    

    # KMeans euclidean
    acc = kmeans(train, query, 'euclidean')
    print('KMeans euclidean: ',acc)

    # KMeans cosim
    acc = kmeans(train, query, 'cosim')
    print('KMeans cosim: ',acc)
    
    # Soft KMeans
    acc = soft_kmeans(train, query, 'euclidean')
    print('Soft Kmeans euclidean: ', acc)

    
if __name__ == "__main__":
    main()
    