import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

def learning_curve_plotter(clf, x, y, metric="f1", train_sizes=np.linspace(0.1, 1.0, 10), save_name="lr.png"):
    
    tr_sizes, tr_scores, test_scores = learning_curve(
        estimator=clf,
        scoring=metric,
        X=x,
        y=y,
        train_sizes=train_sizes
        )
    
    tr_avg = np.mean(tr_scores, axis=1)
    tr_std = np.std(tr_scores, axis=1)
    test_avg = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(tr_sizes, tr_avg, marker='o', label="training")
    plt.fill_between(tr_sizes, tr_avg+tr_std, tr_avg-tr_std, color='blue', alpha=0.1)
    plt.plot(tr_sizes, test_avg, marker='v', label="validation")
    plt.fill_between(tr_sizes, test_avg+test_std, test_avg-test_std, color='orange', alpha=0.1)
    plt.grid()
    plt.xlabel("# of training samples")
    plt.ylabel("F1 Score")
    plt.legend()
    # plt.show()
    plt.savefig(os.path.join("fig", save_name))
    plt.close()

def model_param_curve(grid_search_result_df, x_label, params, param_name, save_fig_name):
    
    x_range=params[param_name]
    x_label=x_label
    y_label="f1_micro"
    title="Default of Credit Card"
    
    mean_train = grid_search_result_df["mean_train_score"]
    mean_test = grid_search_result_df["mean_test_score"]
    std_train = grid_search_result_df["std_train_score"]
    std_test = grid_search_result_df["std_test_score"]
    
    #Plotting
    plt.grid()
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    plt.plot(x_range, mean_train, marker='o', label='train', color=f"C{1}")
    plt.fill_between(x_range, mean_train+std_train,
            mean_train-std_train, color=f'C{1}', alpha=0.1)
    plt.plot(x_range, mean_test, marker='v', label='valid', color=f"C{2}")
    plt.fill_between(x_range, mean_test+std_test,
             mean_test-std_test, color=f"C{2}", alpha=0.1)
    plt.legend()
    plt.savefig(os.path.join("fig", save_fig_name))
    plt.show()
    plt.close()

def score(X_train, y_train, X_test, y_test, model):
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"score - train: {train_score}, test: {test_score}")

# def metrics(true_label, predict_label):
#     roc = roc_auc_score(true_label, predict_label)
#     f1 = f1_score(true_label, predict_label)
#     print(f"test metrics - roc: {roc}, f1: {f1}")
#     return roc, f1

def metrics(X_train, y_train, X_test, y_test, model):
    train_roc = roc_auc_score(y_train, model.predict(X_train))
    train_f1 = f1_score(y_train, model.predict(X_train)) 

    test_roc = roc_auc_score(y_test, model.predict(X_test))
    test_f1 = f1_score(y_test, model.predict(X_test))
    print(f"train / test metrics - tr_roc: {train_roc}, tr_f1: {train_f1}, test_roc: {test_roc}, test_f1: {test_f1}")
    return train_roc, train_f1, test_roc, test_f1