import pandas as pd
import numpy as np
import dataset
from sklearn.model_selection import train_test_split, GridSearchCV #データ分割用,グリッドサーチ
from sklearn.ensemble import RandomForestClassifier #ランダムフォレスト

x_train, x_test, y_train, y_test = dataset.get_dataset()

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
