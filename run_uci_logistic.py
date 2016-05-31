# ch03
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import import_data

# Fettermania libraries 
import plot_decision_regions

# === SECTION: get data ===




# df['is_match'].apply(lambda x: 1.0 if x else 0.0)


# TODO start here 
#df = pd.read_csv("/Users/fettermania/Desktop/Projects/Project_amperity/uci-test/resources/uci/block_1.csv")
#iris = datasets.load_iris()
# Fettermania: why is [2,3] better than all the features?
# X = iris.data[:, [2, 3]]
# y = iris.target

df = import_data.df_from_known_csvs()
import_data.dataframe_clean_columns_in_place(df)
X = df[import_data.feature_columns_from_df(df)]
y = df[import_data.target_column_from_df(df)]

# FETTERMANIA: TPlay
# X = X[['cmp_fname_c1', 'cmp_fname_c2', 'cmp_lname_c1', 'cmp_lname_c2', 'cmp_sex', 'cmp_bd', 'cmp_bm', 'cmp_by', 'cmp_plz', 'cmp_fname_c1_is_null', 'cmp_fname_c2_is_null', 'cmp_lname_c2_is_null', 'cmp_bd_is_null', 'cmp_bm_is_null', 'cmp_by_is_null', 'cmp_plz_is_null']]
X = X[[
'cmp_fname_c1',
 'cmp_fname_c2', 'cmp_lname_c1', 'cmp_lname_c2', 'cmp_sex', 'cmp_bd', 'cmp_bm', 'cmp_by', 'cmp_plz', 'cmp_fname_c1_is_null', 'cmp_fname_c2_is_null', 'cmp_lname_c2_is_null', 'cmp_bd_is_null', 'cmp_bm_is_null', 'cmp_by_is_null', 'cmp_plz_is_null']]
X = X[2090:2100]

print('Class labels:', np.unique(y))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.99, random_state=0)

# === SECTION: Preprocess data ===

# from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# sc.fit(X_train)
# X_train_std = sc.transform(X_train)
# X_test_std = sc.transform(X_test)

X_train_std = X_train
X_test_std = X_test

# === SECTION: Train model ===

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

y_pred = lr.predict(X_test_std)

import sklearn.metrics
print('Accuracy: %.2f' % sklearn.metrics.accuracy_score(y_test, y_pred))
print('Precision: %.2f' % sklearn.metrics.precision_score(y_test, y_pred))
print('Recall: %.2f' % sklearn.metrics.recall_score(y_test, y_pred))

# # FGettermania: predict one
# print('Probs for one prediction:')
# # print(lr.predict_proba(X_test_std[0,:]))

# plot_decision_regions.plot_decision_regions(X_combined_std, y_combined, 
#                       classifier=lr, test_idx=range(105,150))
# plt.xlabel('petal length [standardized]')
# plt.ylabel('petal width [standardized]')
# plt.legend(loc='upper left')
# plt.tight_layout()
# # plt.savefig('./figures/logistic_regression.png', dpi=300)
# plt.show()

# # === SECTION: Plotting one of the two weights vs. regularization C

# weights, params = [], []
# for c in np.arange(-5, 5):
#     lr = LogisticRegression(C=10**c, random_state=0)
#     lr.fit(X_train_std, y_train)
#     weights.append(lr.coef_[1]) 
#     params.append(10**c)

# weights = np.array(weights)
# plt.plot(params, weights[:, 0], 
#          label='petal length')
# plt.plot(params, weights[:, 1], linestyle='--', 
#          label='petal width')
# plt.ylabel('weight coefficient')
# plt.xlabel('C')
# plt.legend(loc='upper left')
# plt.xscale('log')
# # plt.savefig('./figures/regression_path.png', dpi=300)
# plt.show()


