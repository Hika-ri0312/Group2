from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import dataset

x_train, x_test, y_train, y_test = dataset.get_dataset()

tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']}
    ]

score = 'f1'
clf = GridSearchCV(
    svm.LinearSVC(), # 識別器
    tuned_parameters, # 最適化したいパラメータセット 
    cv=5, # 交差検定の回数
    scoring='%s_weighted' % score ) # モデルの評価関数の指定

clf.fit(x_train, y_train)

clf.grid_scores_

clf.best_params_

model = svm.LinearSVC()

model.fit(x_train, y_train)

pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test, pred_test)
print('テストデータに対する正解率： %.2f' % accuracy_test)