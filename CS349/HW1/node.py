class Node:
    def __init__(self):
        self.label = None
        self.attribute = None
        self.entropy = 1
        self.children = {}

    def change_label(self, new_label):
        self.label = new_label

    def change_attribute(self, new_attribute):
        self.attribute = new_attribute

    def add_child(self, value, child):
        self.children[value] = child

    def change_entropy(self, new_entropy):
        self.entropy = new_entropy
