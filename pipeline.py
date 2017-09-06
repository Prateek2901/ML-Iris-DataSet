from sklearn import datasets
iris = datasets.load_iris()

X = iris.data
Y = iris.target

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=.5)

#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()

from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()
my_classifier.fit(X_train,Y_train)

predict = my_classifier.predict(X_test)
#print(predict)

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test,predict))
