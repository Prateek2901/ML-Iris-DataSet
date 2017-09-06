import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0,50,100]

# training data
train_target = np.delete(iris.target,test_idx)
train_data = np.delete(iris.data,test_idx,axis=0)

#print (train_target)
#print (train_data)

#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]
#print (test_target)
#print (test_data)

clf = tree.DecisionTreeClassifier()
clf.fit(train_data,train_target)

print (iris.target_names[test_target])

result = clf.predict(test_data)
print (iris.target_names[result])

from sklearn.metrics import accuracy_score
print(accuracy_score(test_target,result))
