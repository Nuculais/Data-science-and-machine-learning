# -*- coding: utf-8 -*-
"""
Pink Programming ML Workshop
Classification (Banana&Pear).
"""

from sklearn import datasets
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn import tree

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

import pydotplus


#The dataset
Data = datasets.load_iris()

classi = tree.DecisionTreeClassifier()

#cross_val_score(classi, Data.data, data.target, cv=0)

clf= classi.fit(Data.data, Data.target)

prediction = clf.predict(Data.data [:1, :])

print(prediction)

"""
#with open("Data.dot", 'w') as f:
  #  f = tree.export_graphviz(clf, out_file=f)
dot_data = tree.export_graphviz(clf, out_file=None)
gra = pydotplus.graph_from_dot_data(dot_data)
gra.write_pdf("iris.pdf")
"""

"""

X_train, X_test, y_train, y_test = train_test_split(Data.data, Data.target)

#Testar olika modeller


model = LogisticRegression()

model.fit(X_train, y_train)

predicted_test = model.predict(X_test)
predicted_train = model.predict(X_train)

print("Printing training performance")
print(metrics.classification_report(y_train, predicted_train))
print(metrics.confusion_matrix(y_train, predicted_train))

print("")
print("Printing testing performance")
print(metrics.classification_report(y_test, predicted_test))
print(metrics.confusion_matrix(y_test, predicted_test))

"""