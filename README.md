# Decision Tree

Decision Tree is a Python 2.7 application, mainly to classify the Gazelle.com dataset from [Kdd Cup 2000](http://www.kdd.org/kdd-cup/view/kdd-cup-2000), to determine whether a user exits from the web site or continue browsing.

Decision tree is recursively constructed using C4.5 algorithm. Pruning applied when growing the tree using the chi-square statistical method. Samples with missing values are distributed to all children with diminished weights.

## Documentation

### DecisionTree class

#### growTree function

Recursively constructs the Decision Tree. Calculates the entropy for the dataset it receives. If there are no samples or the entropy value is 0, it returns a leaf. If the entropy value is greater than 0, it calculates the information gain values and selects the attribute with the highest information gain value.

Samples with missing value on the selected attribute are distributed to all children with diminished weights.

### Node class

## Dependencies
liac-arff library is required to read ARFF files.
timeit library is required to calculate the elapsed time
scipy.stats.chi2 function is required to calculate the chi2 value.
