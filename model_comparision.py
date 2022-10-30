# ------------------------------------------------------------------------------------------------
# PART: library dependencies: -- sklearn, tenserflow, numpy
# import required packages
import matplotlib.pyplot as plt
import random
import numpy as np
# import datasets, classifiers and performance metrics
from sklearn import datasets, metrics, tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils import preprocess_digits, data_viz, train_dev_test_split, h_param_tuning, get_all_h_params_comb, train_save_model
from joblib import dump,load
# ------------------------------------------------------------------------------------------------


# 1. set the range of hyperparameters
gamma_list  = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]


params = {}
params['gamma'] = gamma_list
params['C'] = c_list

# ------------------------------------------------------------------------------------------------

# 2. for every combiation of hyper parameter values
h_param_comb = get_all_h_params_comb(params)

assert len(h_param_comb) == len(gamma_list)*len(c_list)

def h_param():
    return h_param_comb



train_frac = 0.7
test_frac = 0.15
dev_frac = 0.15

# ------------------------------------------------------------------------------------------------

# PART: -load dataset --data from files, csv, tsv, json, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)


del digits


svm_acc = []
tree_acc = []

for i in range(5):

    X_train, y_train, X_dev, y_dev, X_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac, 1 - (train_frac + dev_frac)
    )

    model_path, clf = train_save_model(X_train, y_train, X_dev, y_dev, None, h_param_comb)
    
    tree_clf = tree.DecisionTreeClassifier()
    tree_clf = tree_clf.fit(X_train, y_train)

    tree_pre = tree_clf.predict(X_test)
    best_model = load(model_path)
    predicted= best_model.predict(X_test)
    _, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
    for ax, image, prediction in zip(axes, X_test, predicted):
        ax.set_axis_off()
        image = image.reshape(8, 8)
        ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
        ax.set_title(f"Prediction: {prediction}")

    svm_acc.append(accuracy_score(y_test, predicted))
    tree_acc.append(accuracy_score(y_test, tree_pre))


# ----------------------------------------------------------------------------
# # PART: Compute evaluation Matrics 
# 4. report the best set accuracy with that best model.

def display(svm_acc, tree_acc):

    print(f'SVM: {svm_acc}')
    print(f'TREE: {tree_acc}')
    print()

    

def calculate_mean(svm_acc, tree_acc):
    svm_acc = np.array(svm_acc)
    tree_acc = np.array(tree_acc)

    svm_mean = np.mean(svm_acc)
    tree_mean = np.mean(tree_acc)

    print(f"SVM Mean: {svm_mean} \t Tree mean: {tree_mean}")

    return [svm_mean, tree_mean]


def calculate_var(svm_acc, tree_acc):
    svm_acc = np.array(svm_acc)
    tree_acc = np.array(tree_acc)

    svm_variance = np.var(svm_acc)
    tree_variance = np.var(tree_acc)

    print(f'SVM Variance: {svm_variance} \t \t Tree Variance: {tree_variance}')


# ----------------------------------------------------------------------------
# Check which one is best
def checkBest(svm_mean, tree_mean):
    if svm_mean > tree_mean:
        print("svm is best")
        
    else:
        print("tree is best")
    print()


# ----------------------------------------------------------------------------



display(svm_acc, tree_acc)
svm_mean,tree_mean = calculate_mean(svm_acc, tree_acc)
calculate_var(svm_acc, tree_acc)
print()
checkBest(svm_mean, tree_mean)
