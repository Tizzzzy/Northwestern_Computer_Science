from sklearn.datasets import make_blobs
X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
print(X.dtype)
print(y.dtype)

'''
    k = 10
    labels = []
    train_no_label = []
    train_label = []
    query_no_label = []
    query_label = []
    for label, value in train:
        train_no_label.append(value)
        train_label.append(label)
    for label, value in query:
        query_no_label.append(value)
        query_label.append(label)
    
    # print(len(train_no_label), len(query_no_label))
    centroids = random.sample(range(len(train)), k)
    # print(centroids)

    train_no_label = np.array(train_no_label, dtype=np.int32)
    query_no_label = np.array(query_no_label, dtype=np.int32)

    max_iteration = 100
    for iteration in range(max_iteration):
        cluster = []
        for test in query_no_label:
            # test = np.array(test, dtype=np.int32)
            if metric == 'euclidean':
                distance = [euclidean(train_no_label[i], test) for i in centroids]
            elif metric == 'cosim':
                distance = [cosim(train_no_label[i], test) for i in centroids]
                distance = [1 - dist for dist in distance]
          
            nearest = np.argmin(distance)
            cluster.append(nearest)
'''