from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import dataset

df = dataset.get_dataset()
x_train, x_test, y_train, y_test = train_test_split(df["images"], df["painters"], test_split=0.3 , random_state=0)

tuned_parameters = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']}
    ]

score = 'f1'
clf = GridSearchCV(
    svm.LinearSVC(), #識別器
    tuned_parameters, #最適化したいパラメータセット 
    cv=5, #交差検定の回数
    scoring='%s_weighted' % score ) #モデルの評価関数の指定

clf.fit(x_train, y_train)

clf.grid_scores_

clf.best_params_