import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import import_data


print('Importing data from block_*.csv...')
df = import_data.df_from_known_csvs()
import_data.dataframe_clean_columns_in_place(df)

X = df[import_data.feature_columns_from_df(df)]
y = df[import_data.target_column_from_df(df)]

def print_scores(y_test, y_pred):
  print('Accuracy: %.6f' % sklearn.metrics.accuracy_score(y_test, y_pred))
  print('Precision: %.6f' % sklearn.metrics.precision_score(y_test, y_pred))
  print('Recall: %.6f' % sklearn.metrics.recall_score(y_test, y_pred))
  print('F1 score: %.6f' % sklearn.metrics.f1_score(y_test, y_pred))

  cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
  print('Confusion matrix')
  print(cm)


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=0)

import sklearn.metrics
from sklearn.linear_model import LogisticRegression
np.set_printoptions(precision=6) # for confusion matrix



# MODEL: Dummy
y_pred = np.zeros(y_test.shape[0])
print('----')
print("Dummy model: Always 0")
print_scores(y_test, y_pred)


# MODEL: Perceptron
from sklearn.linear_model import Perceptron
ppn_eta = 0.1
ppn = Perceptron(n_iter=40, eta0=ppn_eta, random_state=0)
ppn.fit(X_train, y_train)
model = ppn
y_pred = model.predict(X_test)
print('----')
print('Perceptron, eta = %f' % ppn_eta)
print_scores(y_test, y_pred)


# MODEL: Logistic Regression
for exp in range(-5, 6):
  lr = LogisticRegression(C=10**exp, random_state=0)
  lr.fit(X_train, y_train)
  model = lr

  y_pred = model.predict(X_test)

  print('----')
  print('Logistic, C = %f' % (10**exp))
  print_scores(y_test, y_pred)



