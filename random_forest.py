# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import dataset
from sklearn.model_selection import GridSearchCV #データ分割用,グリッドサーチ
from sklearn.ensemble import RandomForestClassifier #ランダムフォレスト

def random_forest():
    x_train, x_test, y_train, y_test = dataset.load_dataset()

    parameters = {
        'n_estimators':[10,20,30,50,100,300],           #用意する決定木
        'max_features':('sqrt', 'log2','auto', None),   #ランダムに指定する特徴量の数
        'max_depth':(10,20,30,40,50,None),              #決定木のノード深さの制限
        
    }
    
    #モデルインスタンス
    model  = RandomForestClassifier()

    #ハイパーパラメーターチューニング（グリッドサーチのコンストラクタにモデルと辞書パラメータを指定)
    gridsearch = GridSearchCV(estimator = model,         #モデル
                            param_grid = parameters,   #チューニングするハイパーパラメータ
                            scoring = "accuracy")      #スコアリング

    #演算実行
    gridsearch.fit(x_train, y_train)

    # グリッドサーチの結果から得られた最適なパラメータ候補を確認
    print('Best params: {}'.format(gridsearch.best_params_)) 
    print('Best Score: {}'.format(gridsearch.best_score_))
    
    
    # 最適なハイパーパラメータの組み合わせを用いてモデル再構築
    model = RandomForestClassifier(n_estimators = gridsearch.best_params_['n_estimators'], # 用意する決定木モデルの数
                               max_features = gridsearch.best_params_['max_features'], # ランダムに指定する特徴量の数
                               max_depth    = gridsearch.best_params_['max_depth'],    # 決定木のノード深さの制限値
                               criterion='gini',                                       # 不純度評価指標の種類(ジニ係数）
                               min_samples_leaf = 1,                                   # 1ノードの深さの最小値
                               random_state = 0,                                       # 乱数シード
                              )

    # モデル学習
    model.fit(x_train,y_train)
    
    from sklearn import metrics

    # 評価
    predicted = model.predict(x_test)
    print(metrics.accuracy_score(predicted,y_test))

random_forest()