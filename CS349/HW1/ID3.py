from node import Node
import math
from parse import parse
import random
import copy
from matplotlib import pyplot as plt
from collections import Counter


def get_most_common_label(data: list, attribute: str) -> (str, int):
    """
    Get the most common label within the dataset column of attribute
    """
    d = {}
    for row in data:
        value = row[attribute]
        if value not in d:
            d[value] = 0
        d[value] += 1
    return max(d.items(), key=lambda x: x[1])


def get_entropy(data: list) -> float:
    """
    Get the entropy of given dataset using the label of its 'Class'
    """
    entropy = 0
    num_rows = len(data)
    d = {}
    for row in data:
        if row["Class"] not in d:
            d[row["Class"]] = 0
        d[row["Class"]] += 1

    for v in d.values():
        prob = v / num_rows
        entropy -= prob * math.log2(prob)
    return entropy


def get_entropy_with_split(data: list, attribute: str) -> float:
    """
    Get total entropy of the dataset with the split on given attribute
    """
    d = {}
    for row in data:
        if row[attribute] not in d:
            d[row[attribute]] = []
        d[row[attribute]].append(row)

    entropy = 0
    for v in d.values():
        entropy += get_entropy(v) * len(v) / len(data)
    return entropy


def filter_data(data: list, attribute: str, label: str):
    """
    Filter out the given dataset s.t.
        in the returned dataset, all data have label == given label in given attribute value
    """
    l = []
    for row in data:
        if row[attribute] == label:
            l.append(row)
    return l


def ID3_helper(data: list, attributes: list, target=0) -> Node:
    """
    Recursive ID3 function
    """
    # print(data)
    node = Node()
    most_common_class, most_common_class_appear_times = get_most_common_label(data, "Class")
    node.change_label(most_common_class)
    total_num_rows = len(data)

    if most_common_class_appear_times == total_num_rows:
        # print("reaching all 0/1 situation")
        return node

    if len(attributes) == 0:
        # print("no more attributes")
        return node

    entropies = {
        attribute: get_entropy_with_split(data, attribute) for attribute in attributes
    }
    attribute_entropy_pair: (str, float) = min(entropies.items(), key=lambda x: x[1])
    best_attribute = attribute_entropy_pair[0]
    best_entropy = attribute_entropy_pair[1]

    node.change_entropy(best_entropy)
    node.change_attribute(best_attribute)

    # print(best_attribute, entropies)

    new_attributes = attributes.copy()
    new_attributes.remove(best_attribute)

    for different_value in set([row[best_attribute] for row in data]):
        new_data = filter_data(data, best_attribute, different_value)

        if len(new_data) != 0:
            # print("branch: " + str(different_value))
            node.add_child(
                different_value, ID3_helper(new_data, new_attributes)
            )
        else:
            # print("empty branch: " + str(different_value))
            new_node = Node()
            common_class = get_most_common_label(data, "Class")[0]
            new_node.change_label(common_class)
            node.add_child(common_class, new_node)

    return node


def clean_data(data: list) -> list:
    """
    Given dataset, when see a attribute value == "?", change it to the mode of this attribute value set.
    """
    attributes = list(data[0].keys())
    attributes.remove("Class")
    for attribute in attributes:

        d = {}

        non_question_mark_num = 0
        for row in data:
            value = row[attribute]
            if value != "?":
                if value not in d:
                    d[value] = 0
                d[value] += 1
                non_question_mark_num += 1

        mode = max(d.items(), key=lambda x: x[1])

        for row in data:
            value = row[attribute]
            if value == "?":
                row[attribute] = mode[0]

    return data


def ID3(examples: list, default=0) -> Node:
    """
    Prep function for recursive ID3
    """
    attributes = list(examples[0].keys())
    attributes.remove("Class")
    return ID3_helper(clean_data(examples), attributes)


def find_deepest_node(root: Node):
    """
    Takes in a trained tree and a validation set of examples.  Prunes nodes in order
    to improve accuracy on the validation data; the precise pruning strategy is up to you.
    """
    queue = [root]
    last_to_leaf = None
    while queue:
        root = queue.pop(0)
        if len(root.children) != 0:
            last_to_leaf = root
            for _, child in root.children.items():
                queue.append(child)
    return root, last_to_leaf


def prune_by_reduced_error(node: Node, examples: list):
    """
    Prune by reduced error method.
    """
    cur_precision = test(node, examples)
    # print("before", cur_precision)
    cur_node = copy.copy(node)

    # last_to_leaf = find_second_to_last_deepest(cur_node)
    leaf, last_to_leaf = find_deepest_node(cur_node)
    if not last_to_leaf:
        return
    last_to_leaf.children = {}
    after_precision = test(cur_node, examples)
    # print("after", after_precision)
    if after_precision >= cur_precision:
        # print("pruning")
        prune_by_reduced_error(cur_node, examples)
    # else:
    # print("no pruning")

def prune_by_validation_top_to_bottom(node, examples):
    """
    Prune the tree from top to bottom
    """
    childrens = None
    if not node:
        return
    for _, child in node.children.items():
        prune_by_validation2(child, examples)
    if node.children:
        precision_before = test(node, examples)
        childrens = node.children
        node.children = {}
        precision_after = test(node, examples)
        if precision_after < precision_before:
            node.children = childrens
        else:
            node.children = {}    
    

def prune_by_entropy(node: Node, parent_entropy: float):
    """
    Pruning method: Compare the entropy of parent and child,
        if the difference is greater than some threshold, prune the tree
    """
    threshold = 0.1
    node_entropy = node.entropy
    for _, child in node.children.items():
        prune_by_entropy(child, node_entropy)
    if abs(parent_entropy - node_entropy) > threshold:
        node.children = {}


def prune(node: Node, examples: list):
    """
    Caller function to other pruning functions
    """
    prune_by_reduced_error(node, examples)
    prune_by_entropy(node, node.entropy)
    pass


def test(node: Node, examples: list):
    """
    Given a decision tree and unseen datasets, returns accuracy of how well this tree predict
    """
    num_correct = 0
    num_incorrect = 0

    examples = clean_data(examples)

    for example in examples:
        predicted_class = evaluate(node, example)
        if predicted_class == example["Class"]:
            num_correct += 1
        else:
            num_incorrect += 1
    return num_correct / (num_correct + num_incorrect)


def evaluate(node: Node, example):
    """
    Given a decision tree and an example of data, return the predicted class label using the tree
    """
    while len(node.children) != 0:
        attribute = node.attribute
        label = example[attribute]
        if label != "?" and label in node.children:
            node = node.children[label]
        else:
            break
    return node.label


def my_test():
    # data = parse("tennis.data")
    # data = parse("candy.data")
    data = parse("house_votes_84.data")

    # for i in range(100):
    #     ID3_helper(data)
    #ID3(data)

    
def plot_learning_curve(data, training_size):
    data = parse(data)
    training_size.sort()
    pruning_avg = []
    no_pruning_avg = []
    diff = []
    for size in training_size:
        withPruning = []
        withoutPruning = []
        if size > len(data):
            raise ValueError("Input size larger than dataset!")
        sub_data = random.sample(data, size)
        for i in range(100):
            random.shuffle(data)
            train = data[:len(data)//2]
            valid = data[len(data)//2:3*len(data)//4]
            test_d = data[3*len(data)//4:]

            tree = ID3(train, 'democrat')
            acc = test(tree, train)
            acc = test(tree, valid)
            acc = test(tree, test_d)

            prune(tree, valid)
            acc = test(tree, train)
            acc = test(tree, valid)
            acc = test(tree, test_d)
            withPruning.append(acc)
            tree = ID3(train+valid, 'democrat')
            acc = test(tree, test_d)
            withoutPruning.append(acc)
        pruning_avg.append(sum(withPruning)/len(withPruning))
        no_pruning_avg.append(sum(withoutPruning)/len(withoutPruning))
        diff.append(sum(withPruning)/len(withPruning) - sum(withoutPruning)/len(withoutPruning))
    
    plt.figure(figsize=(12, 6))  # Adjust the figure size

    # Create the first subplot for current data
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    #plt.scatter(training_size, pruning_avg, label="With Pruning")
    #plt.scatter(training_size, no_pruning_avg, label="Without Pruning")
    plt.plot(training_size, pruning_avg, linestyle='-', label="With Pruning")
    plt.plot(training_size, no_pruning_avg, linestyle='-', label="Without Pruning")
    plt.legend()
    plt.xlabel("Training Size")
    plt.ylabel("Average Accuracy")
    plt.title("Current Data")

    plt.subplot(1, 2, 2)  
    #plt.scatter(training_size, diff, label="Difference")
    plt.plot(training_size, diff, linestyle='-', label="Difference")
    plt.legend()
    plt.xlabel("Training Size")
    plt.ylabel("Difference")
    plt.title("Difference Data")

    plt.tight_layout()
    plt.savefig('learning_curve.pdf')
    plt.show()

def test_random_forest(trees, examples):
    num_correct = 0
    num_incorrect = 0

    examples = clean_data(examples)

    for example in examples:
        all_predicted_class = []
        ans = example['Class']
        for tree in trees:
            predicted_class = evaluate(tree, example)
            all_predicted_class.append(predicted_class)
        # majority voting
        prediction = most_frequent_element(all_predicted_class)
        if ans in prediction:
            num_correct += 1
        else:
            num_incorrect += 1
    return num_correct / (num_correct + num_incorrect)


def most_frequent_element(lst):
    # Use Counter to count occurrences of each element
    counts = Counter(lst)
    
    # Find the most common element(s) and their count(s)
    most_common = counts.most_common()
    
    # If you want to handle multiple elements with the same highest count,
    # you can create a list of them.
    most_frequent_elements = [item for item, count in most_common if count == most_common[0][1]]
    
    return most_frequent_elements

if __name__ == "__main__":
    # Input of your choice
    size = [i for i in range(10, 30)]
    plot_learning_curve("house_votes_84.data", size)
