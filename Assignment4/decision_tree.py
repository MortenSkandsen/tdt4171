
from random import randint
from math import log
from classes import Node, Example


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
        #best_attribute = argmax(expected_information_gain, attributes, examples)
        best_attribute = random_importance(attributes)
        attributes.remove(best_attribute)
        #print("Best attribute: ", best_attribute)
        root_node = create_tree(best_attribute)
        print("Splitting attribute ", best_attribute)
        for value in [True, False]:
            # Create a subtree for each of the values of the attribute
            exs = [ex for ex in examples if ex.get_attr(best_attribute) == value]
            subtree = decision_tree_learning(exs, attributes, examples)
            root_node.add_subtree(value, subtree)
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


def goal_entropy(examples):
    """
    Takes in a list of examples
    The goal entropy is defined as the entropy of
    B(p/(p+n)), where p is the positive examples and n the negatives
    """
    if not examples:
        return 0
    # Here we are using the trick that summing over booleans gives 1's for True and 0 for False
    num_positives = sum([example.classification for example in examples])
    # Note: If you are running python 2.x, use float(num_positives) in the fraction below
    pos_ratio = num_positives/len(examples)
    return boolean_entropy(pos_ratio)

def boolean_entropy(prob):
    # If we recieve probability that are 1 or 0, the entropy is 0
    if prob in [0, 1]:
        return 0
    # The boolean entropy of the variable p is defined as 
    # B(p) = -(plog(p) + (1-p)log(1-p))
    return -(prob*log(prob, 2) + (1-prob)*log(1-prob, 2))


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
    examples = []
    for line in lines:
        # Since all our lines are 1 or 2's, we will use booleans to represent them (1 is True, 2 is False)
        line = [x=="1" for x in line if x.isdigit()]
        examples.append(Example(line))
    return examples



def create_tree(attribute):
    print("Created node with attribute ", attribute)
    return Node(attribute)


def print_tree(tree, count):
    print('\t'*count+"Node: ", tree)
    children = tree.children
    print('\t'*count+"Children: ", children)
    for key in children.keys():
        node = children[key]
        
        if isinstance(node, Node) and node.has_children():
            print_tree(node, count+1)


def decide(example, tree):
    """
    In: An instance of an example
        A decision tree 
    Out: The decision based on result of decision tree 
    """
    # Find out what attribute we are deciding on
    attribute = tree.attribute

    value = example.get_attr(attribute)
    next_node = tree.get_child(value)
    # Reached a leaf node
    if next_node in [True, False]:
        return next_node
    return decide(example, next_node)



def validate_test_examples():
    train_examples = create_train_examples()

    print("There are %s train examples" % len(train_examples))
    attributes = list(range(7))
    results = []

    test_examples = create_test_examples()
    print("There are %s test examples" % len(test_examples))
    decisions = []
    tree = decision_tree_learning(train_examples, attributes, None)
    print_tree(tree, 0)
    for ex in test_examples:
        decision = decide(ex, tree)
        decisions.append(decision == ex.classification)
    print_accuracy_stats(decisions)
    

def print_accuracy_stats(decisions):
    print("CORRECT DECISIONS:", sum(decisions))
    print("WRONG DECISIONS:", len(decisions)-sum(decisions))
    print("ACCURACY: ", 100*sum(decisions)/ len(decisions), "%")

validate_test_examples()

