from DecisionTree import *
import pandas as pd
from sklearn import model_selection

#header = ['SepalL', 'SepalW', 'PetalL', 'PetalW', 'Class']
#header = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
#df = pd.read_csv('ftp://ftp.ics.uci.edu/pub/machine-learning-databases/iris/iris.data', header=None, names=['SepalL','SepalW','PetalL','PetalW','Class'])
#df = pd.read_csv('ftp://ftp.ics.uci.edu/pub/machine-learning-databases/car/car.data', header=None, names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety'])
#df=pd.read_csv('Cars.csv')
df=pd.read_csv('Iris.csv')
header = list(df.columns.values)
lst = df.values.tolist()
t = build_tree(lst, header)
print_tree(t)

print("********** Leaf nodes ****************")
leaves = getLeafNodes(t)
for leaf in leaves:
    print("id = " + str(leaf.id) + " depth =" + str(leaf.depth))
print("********** Non-leaf nodes ****************")
innerNodes = getInnerNodes(t)
for inner in innerNodes:
    print("id = " + str(inner.id) + " depth =" + str(inner.depth))

trainDF, testDF = model_selection.train_test_split(df, test_size=0.2)
train = trainDF.values.tolist()
test = testDF.values.tolist()

t = build_tree(train, header)
print("*************Tree before pruning*******")
print_tree(t)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))

## TODO: You have to decide on a pruning strategy
# for i in range(2,6):
#     randomItems = randomPruning(innerNodes,i)
#     print("List of nodes going to be pruned")
#     print(randomItems)
#     #t_pruned = prune_tree(t, [26, 11, 5])
#     t_pruned=prune_tree(t, randomItems)
#     print("*************Tree after pruning*******")
#     #print_tree(t_pruned)
#     acc = computeAccuracy(test, t)
#     print("Accuracy on test = " + str(acc))

randomItems = randomPruning(innerNodes)
print("List of nodes going to be pruned")
print(randomItems)
#t_pruned = prune_tree(t, [26, 11, 5])
t_pruned=prune_tree(t, randomItems)
print("*************Tree after pruning*******")
print_tree(t_pruned)
acc = computeAccuracy(test, t)
print("Accuracy on test = " + str(acc))
