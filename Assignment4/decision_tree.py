
from random import randint
from math import log

def decision_tree_learning(examples, attributes, parent_examples):
    # if examples is empty, return plurality_value(parent_examples)
    if not examples: 
        print("No examples left")
        return plurality_value(parent_examples)
   
    # else if all examples have same classification, return classification (since all have same, return the class. of the first)
    elif has_same_classification(examples): 
        print("All examples have same classification! Number of examples: %s" % len(examples))
        return examples[0].classification

    # else if attributes is empty, return plurality_value(examples)
    # this occurs if there is noise in the examples 
    elif not attributes: 
        print("Attributes is emtpy, examples are noisy!")
        return plurality_value(examples)
    
    else:
        # choose best attribute to split on
        best_attribute = argmax(expected_information_gain, attributes, examples)
        print("Attributes:", attributes)
        attributes.remove(best_attribute)
        print("Best attribute: ", best_attribute)
        root_node = create_tree(best_attribute)
        for value in [True, False]:
            # Create a subtree for each of the values of the attribute
            exs = [ex for ex in examples if ex.get_attr(best_attribute)==value]
            subtree = decision_tree_learning(exs, attributes, examples)
            root_node.add_subtree(subtree)
    return root_node

def random_importance(attributes):
    # Returns a random attribute from in param that is to split the trees
    return attributes[randint(0, len(attributes)-1)]

def argmax(function, *args):
    # Returns the attribute that maximizes the given function given the arguments
    # The first argument is expected to be maximized, while the rest are arguments needed for the function
    attributes = args[0]
    other_args = args[1:]
    # Create exp. info gain for each attribute
    attribute_dict = {attribute: function(attribute, *other_args) for attribute in attributes}
    # Choose the best attribute
    return max(attribute_dict, key= lambda k: attribute_dict[k])

def expected_information_gain(attribute, examples, *args):
    print("Attribute: ", attribute)
    print("Examples: ", examples)
    # This is one of the candidates for the importance function
    # The information gain is defined as the reduction in entropy, 
    # that is Gain(A) = B(p/(p+n)) - Remainder(A)
    # We are using varargs so our argmax function can be as general as possible, but arent using it in this function
    return goal_entropy(examples) - remainder(attribute, examples)


def remainder(attribute, examples):
    # The remainder of an attribute A with values d1,....dn is
    # sum_d((pk+nk)/(p+n)*B(pk/(pk+nk))
    # for boolean values, we only split A into two.

    # Separate examples in the ones that are True and False for A.
    truthy = [ex for ex in examples if ex.get_attr(attribute)]
    falsy = [ex for ex in examples if not ex.get_attr(attribute)]
    
    remainder = 0
    for subset in [truthy, falsy]:
        remainder += len(subset)/len(examples) * goal_entropy(subset)
    return remainder
    


def has_same_classification(examples):
    # Returns true if all examples have same classification, else False
    classifications = [example.classification for example in examples]
    return True not in classifications or False not in classifications

def plurality_value(examples):
    # Returns the classification that mosts of the examples have
    # Since we are dealing with boolean values, we just check if the majority of examples classification are true
    return sum([example.classification for example in examples]) > len(examples)//2

def read_file_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def create_train_examples():
    lines = read_file_lines('data/training.txt')
    return create_example_classes(lines)
    
def create_test_examples():
    lines = read_file_lines('data/test.txt')
    return create_example_classes(lines)

def create_example_classes(lines):
    # Iterate over all examples, create class instances
    for line in lines:
        # Since all our lines are 1 or 2's, we will use booleans to represent them (1 is True, 2 is False)
        line = [x=="1" for x in line if x.isdigit()]
        examples.append(Example(line))
    return examples

class Example:
    attributes = []
    classification = None
    def __init__(self, values):
        # In: A list of values
        # All but the last value are values of attributes
        # The last value is by definition the classification of the given Example
        if(len(values) != 8): raise ValueError("Incorrect format of value list!")
        self.attributes = values[:7]
        self.classification = values[-1]

    def get_attr(self, index):
        # Returns attribute value with the given index
        return self.attributes[index]

    def __repr__(self):
        return "Example: %s" % str(self.classification)

class Node:
    def __init__(self, children, attribute):
        # Children is a list of Nodes
        self.children = children
        # Attribute is what attribute the node represents
        self.attribute = attribute
        # What value of attribute was chosen to get to this node?
        #self.branch_value = branch_value
    
    def add_subtree(self, subtree):
        self.children.append(subtree)
    
    def __repr__(self):
        return "Node: {}".format(self.attribute)

def create_tree(attribute):
    print("Created node with attribute ", attribute)
    return Node([], attribute)


def goal_entropy(examples):
    """
    Takes in a list of examples
    The goal entropy is defined as the entropy of
    B(p/(p+n)), where p is the positive examples and n the negatives
    """
    
    # Here we are using the trick that summing over booleans gives 1's for True and 0 for False
    num_positives = sum([example.classification for example in examples])
    # Note: If you are running python 2.x, use float(num_positives) in the fraction below
    pos_ratio = num_positives/len(examples)
    return boolean_entropy(pos_ratio)

def boolean_entropy(prob):
    print("Probability: ", prob)
    # If we recieve sets that are 
    if prob in [0, 1]:
        return 0
    # The boolean entropy of the variable p is defined as 
    # B(p) = -(plog(p) + (1-p)log(1-p))
    return -(prob*log(prob, 2) + (1-prob)*log(1-prob, 2))

def print_tree(tree):
    print(tree)
    children = tree.children
    print(children)
    for node in children:
        if type(node) == Node and node.children:
            print_tree(node)


examples = create_train_examples()

print("There are %s examples" % len(examples))
attributes = list(range(len(examples[0].attributes)))
print("Number of attributes: ", len(attributes))

tree = decision_tree_learning(examples, attributes , None)
print_tree(tree)