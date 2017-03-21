class Example:
    attributes = []
    classification = None
    def __init__(self, values):
        # In: A list of values
        # All but the last value are values of attributes
        # The last value is by definition the classification of the given Example
        if len(values) != 8:
            raise ValueError("Incorrect format of value list!")
        self.attributes = values[:7]
        self.classification = values[-1]

    def get_attr(self, index):
        # Returns attribute value with the given index
        return self.attributes[index]

    def __repr__(self):
        return "Example with class.: %s" % str(self.classification)
    
    def __eq__(self, other):
        return all(self.attributes[i] == other.attributes[i] for i in range(len(self.attributes)))
class Node:
    def __init__(self, attribute):
        # Children is a dictionary of Nodes where key is the chosen attribute, and value is the next node
        self.children = {}
        # Attribute is what attribute the node represents
        self.attribute = attribute
        
    
    def add_subtree(self, key, subtree):
        self.children[key] = subtree
    
    def has_children(self):
        return len(self.children.keys()) > 0
    
    def get_child(self, key):
        return self.children[key]

    def __repr__(self):
        return "Node: {}".format(self.attribute)