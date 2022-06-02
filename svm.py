from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
import dataset

df = dataset.get_dataset()
x_train, x_test, y_train, y_test = train_test_split(df["images"], df["painters"], test_split=0.3 , random_state=0)

model = svm.LinearSVC()

model.fit(x_train, y_train)

pred_test = model.predict(x_test)
accuracy_test = accuracy_score(y_test, pred_test)
print('テストデータに対する正解率： %.2f' % accuracy_test)