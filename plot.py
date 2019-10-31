import numpy as np
import matplotlib.pyplot as plt
from main import *
from sklearn.metrics import roc_curve
clf_fpr, clf_tpr, clf_thresholds = roc_curve(y_test, clf.predict_proba(X_test)[:,1])

plt.figure()

# 决策树 ROC
plt.plot(clf_fpr, clf_tpr, label='Decision Tree (area = %0.2f)' % clf_roc_auc)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Graph')
plt.legend(loc="lower right")
plt.show()

# 特征重要性
importances = clf.feature_importances_
feat_names = df.drop(['left'],axis=1).columns
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("Feature importances by Decision Tree")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.xticks(range(len(indices)), feat_names[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()