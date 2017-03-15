


def decision_tree_learning(examples, attributes, parent_examples):
    # if examples is empty, return plurality_value(parent_examples)
    if not examples: return plurality_value(parent_examples)
   

    # else if all examples have same classification, return classification (since all have same, return the class. of the first)
    elif has_same_classification(examples): return examples[0].classification

    # else if attributes is empty, return plurality_value(examples)
    # this occurs if there is noise in the examples 
    elif not attributes: return plurality_value(examples)
    
    # else, choose best attribute to split on, then create new trees
    # A = argmax_a Importance(a, examples) 
    pass

def expected_information_gain(attribute, examples):
    # The information gain is defined as the reduction in entropy, 
    # that is Gain(A) = B(p/(p+n)) - Remainder(A)

def remainder(attributes):
    # The remainder of an attribute A with values d1,....dn is
    # sum_d((pk+nk)/(p+n)*B(pk/(pk+nk))

def has_same_classification(examples):
    # Returns true if all examples have same classification, else False
    sum_classification = sum([example.classification for example in examples])
    return sum_classification in [0, len(examples)]

def plurality_value(examples):
    # Returns the classification that mosts of the examples have
    # Since we are dealing with boolean values, we just check if the majority of examples classification are true
    return sum([example.classification for example in examples]) > len(examples)//2

def read_file_lines(filename):
    with open(filename, 'r') as f:
        return f.readlines()

def create_train_examples():
    lines = read_file_lines('data/training.txt')
    
    examples = []
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
    

def goal_entropy(examples):
    # Takes in a list of examples
    # The goal entropy is defined as the entropy of
    # B(p/(p+n)), where p is the positive examples and n the negatives

    # Here we are using the trick that summing over booleans gives 1's for True and 0 for False
    num_positives = sum([example.classification for example in examples])
    
    # Note: If you are running python 2.x, use float(num_positives) in the fraction below
    return boolean_entropy(num_positives/len(examples))

def boolean_entropy(prob):
    # The boolean entropy of the variable p is defined as 
    # B(p) = -(plog(p) + (1-p)log(1-p))
    from math import log
    return -(prob*log(prob, 2) + (1-prob)*log(1-prob, 2))

examples = create_train_examples()
