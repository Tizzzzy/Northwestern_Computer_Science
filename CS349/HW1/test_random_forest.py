import ID3, parse, random
from node import Node


def testRandomForest(inFile):
    random_forest = []
    data = parse.parse(inFile)

    for i in range(100):
        train = data[:len(data)//2]
        valid = data[len(data)//2:3*len(data)//4]
        test = data[3*len(data)//4:]
        random.shuffle(data)
        tree = ID3.ID3(train, 'democrat')
        # tree = ID3.ID3(train, 0)
        random_forest.append(tree)

    acc = ID3.test_random_forest(random_forest, train)
    print("training accuracy: ",acc)
    acc = ID3.test_random_forest(random_forest, valid)
    print("validation accuracy: ",acc)
    acc = ID3.test_random_forest(random_forest, test)
    print("test accuracy: ",acc)



if __name__ == "__main__":
    testRandomForest("./candy.data")
