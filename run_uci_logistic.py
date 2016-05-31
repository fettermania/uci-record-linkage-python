import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import import_data


df = import_data.df_from_known_csvs()

import_data.dataframe_clean_columns_in_place(df)
X = df[import_data.feature_columns_from_df(df)]
y = df[import_data.target_column_from_df(df)]

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)



# from sklearn.linear_model import Perceptron
# ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
# ppn.fit(X_train, y_train)
# model = ppn

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train, y_train)
model = lr

y_pred = model.predict(X_test)

import sklearn.metrics
print('Accuracy: %.6f' % sklearn.metrics.accuracy_score(y_test, y_pred))
print('Precision: %.6f' % sklearn.metrics.precision_score(y_test, y_pred))
print('Recall: %.6f' % sklearn.metrics.recall_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=6)
print('Confusion matrix')
print(cm)


