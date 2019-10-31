from decribe import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score

clf = DecisionTreeClassifier(criterion='entropy')
# print(X_train.head())

clf = DecisionTreeClassifier(
    criterion='entropy',
    min_weight_fraction_leaf=0.01
    )
clf = clf.fit(X_train,y_train)
clf_roc_auc = roc_auc_score(y_test, clf.predict(X_test))
print("决策树 AUC = %2.2f" % clf_roc_auc)

