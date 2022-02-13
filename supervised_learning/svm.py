import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
sns.set_style("dark")

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

# Models
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from utils import learning_curve_plotter, model_param_curve, score, metrics

# base param
seed = 712

# Load Dataset
credit_card_path = "../data/UCI_Credit_Card.csv"
df_credit = pd.read_csv(credit_card_path)
# df_credit = df_credit[:100]

edu_converter = (df_credit.EDUCATION == 5) | (df_credit.EDUCATION == 6) | (df_credit.EDUCATION == 0)
df_credit.loc[edu_converter, 'EDUCATION'] = 4
df_credit["MARRIAGE"] = df_credit["MARRIAGE"].replace(0, 3)

target = 'default.payment.next.month'
features = [  'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 
                'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6', 
                'BILL_AMT1','BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6',
                'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6']

# features = [  'LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE' ]


train_df, test_df = train_test_split(df_credit, test_size=0.2, random_state=712)
X_train, y_train = train_df[features], train_df[target]
X_test, y_test = test_df[features], test_df[target]


# # with scale
sc = StandardScaler()
sc.fit(X_train)

X_train = sc.transform(X_train)
X_test = sc.transform(X_test)

svm_params = {
    "kernel":["linear", "rbf"],
    "C": [0.001, 0.01, 0.1, 1, 10]
}


# svm_params = {
#     "kernel":["linear"],
#     "C": [0.1]
# }

def model_param_curve_multi(grid_search_result_df, x_label, params, param_name, save_fig_name):

    x_range=params[param_name]
    x_label=x_label
    y_label="f1_micro"
    title="Default of Credit Card"
        
    linear_df = grid_search_result_df[svm_grid_search_df['param_kernel'].str.contains('linear')]
    rbf_df = grid_search_result_df[svm_grid_search_df['param_kernel'].str.contains('rbf')]

    linear_mean_train = linear_df["mean_train_score"]
    linear_mean_test = linear_df["mean_test_score"]
    linear_std_train = linear_df["std_train_score"]
    linear_std_test = linear_df["std_test_score"]
    
    rbf_mean_train = rbf_df["mean_train_score"]
    rbf_mean_test = rbf_df["mean_test_score"]
    rbf_std_train = rbf_df["std_train_score"]
    rbf_std_test = rbf_df["std_test_score"]

    #Plotting
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.plot(x_range, linear_mean_train, marker='o', label='linear train', color=f"C{1}")
    plt.fill_between(x_range, linear_mean_train+linear_std_train,
            linear_mean_train-linear_std_train, color=f'C{1}', alpha=0.1)
    plt.plot(x_range, linear_mean_test, marker='o', label='linear valid', color=f"C{2}")
    plt.fill_between(x_range, linear_mean_test+linear_std_test,
             linear_mean_test-linear_std_test, color=f"C{2}", alpha=0.1)
    plt.plot(x_range, rbf_mean_train, marker='v', label='rbf train', color=f"C{3}")
    plt.fill_between(x_range, rbf_mean_train+rbf_std_train,
             rbf_mean_train-rbf_std_train, color=f"C{3}", alpha=0.1)
    plt.plot(x_range, rbf_mean_test, marker='v', label='rbf valid', color=f"C{4}")
    plt.fill_between(x_range, rbf_mean_test+rbf_std_test,
             rbf_mean_test-rbf_std_test, color=f"C{4}", alpha=0.1)

    plt.legend()
    plt.savefig(save_fig_name)
    plt.show()


# svm_clf = svm.SVC(random_state=seed, verbose=False)
# svm_grid_search = GridSearchCV(estimator=svm_clf, param_grid=svm_params,
#                            scoring='f1_micro', return_train_score=True,
#                            verbose=3, n_jobs=-1)
# svm_grid_search.fit(X_train, y_train)

# svm_grid_search.cv_results_.keys()
# svm_grid_search_df = pd.DataFrame(svm_grid_search.cv_results_)
# svm_grid_search_df.sort_values(by="rank_test_score")

# linear_df = svm_grid_search_df[svm_grid_search_df['param_kernel'].str.contains('linear')]
# rbf_df = svm_grid_search_df[svm_grid_search_df['param_kernel'].str.contains('rbf')]

# best_params = svm_grid_search_df["params"][np.argmax(svm_grid_search_df['mean_test_score'])]
# best_score = np.max(svm_grid_search_df['mean_test_score'])
# print(best_params)
# model_param_curve_multi(svm_grid_search_df, "C value", svm_params, "C", "svm_c.png")

best_svm_clf = svm.SVC(
    random_state=seed,
    C=1,
    kernel='rbf',
    verbose=True
)
best_svm_clf.fit(X_train, y_train)
learning_curve_plotter(best_svm_clf, X_train, y_train, save_name="svm_lr.png")
metrics(X_train, y_train, X_test, y_test, best_svm_clf)



# kernels = ["linear", "rbf"]
# C = [0.1,1,100,1000]

# tr_scores = np.zeros((len(C), len(kernels)))
# test_scores = np.zeros((len(C), len(kernels)))

# for i in range(len(C)):
#     for j in range(len(kernels)):
#         svm_clf = svm.SVC(C=C[i], kernel=kernels[j], random_state=seed, verbose=True)
#         svm_clf.fit(X_train, y_train)
#         tr_roc, tr_f1, test_roc, test_f1 = metrics(X_train, y_train, X_test, y_test, svm_clf)
#         tr_scores[i,j] = tr_f1
#         test_scores[i,j] = test_f1
#         print(f"tr_f1: {tr_f1}, test_f1: {test_f1}, C: {C[i]}, kernel: {kernels[j]}")



