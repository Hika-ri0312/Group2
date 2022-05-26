import pandas as pd
import numpy as np
import dataset
from sklearn.model_selection import train_test_split, GridSearchCV #データ分割用,グリッドサーチ
from sklearn.ensemble import RandomForestClassifier #ランダムフォレスト

x_train, x_test, y_train, y_test = train_test_split(x.values, y.values,test_size = 0.3, random_state = 3)

 


