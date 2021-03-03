import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as plt


#Dataset for SD3/IPIP/PVQ
df = pd.read_csv("new_PVQ_new_harry_all_remove.csv", sep = ',')

#Splitting data into train, development, test sets
def train_validate_test_split(df, train_percent=.7, validate_percent=.15, seed=None):
    np.random.seed(seed)
    perm = np.random.permutation(df.index)
    m = len(df.index)
    train_end = int(train_percent * m)
    validate_end = int(validate_percent * m) + train_end
    train = df.loc[perm[:train_end]]
    validate = df.loc[perm[train_end:validate_end]]
    test = df.loc[perm[validate_end:]]
    return train, validate, test

train, validate, test = train_validate_test_split(df)

print(train.head())

#21 features for SD3/IPIP data and 35 features for SD3/IPIP/PVQ data 
X_train = np.array(train.iloc[:, 1:36])
y_train = list(train.iloc[:, 0])
X_develop = np.array(validate.iloc[:, 1:36])
y_develop = list(validate.iloc[:, 0])
X_test = np.array(test.iloc[:, 1:36])
y_test = list(test.iloc[:, 0])


#standardize data
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_develop_std = sc.transform(X_develop)
X_test_std = sc.transform(X_test)

#normalize data
mms = MinMaxScaler()
X_train_norm = mms.fit_transform(X_train)
X_develop_norm = mms.fit_transform(X_develop)
X_test_norm = mms.transform(X_test)


#GRID SEARCH for all 6 models

print('Grid Search Analysis')

#Logistic Regression grid search
param_C = [.0001, .001, .01, 1, 10, 100, 1000, 10000]
param_solver = ['newton-cg', 'sag', 'saga', 'lbfgs']
param_grid = [{'penalty': ['l1'], 'solver': ['saga'], 'C': param_C}, \
               {'penalty': ['l2'], 'solver': param_solver, 'C': param_C}]
lr = LogisticRegression(multi_class = 'ovr', max_iter = 5000, random_state = 1)
gs = GridSearchCV(estimator = lr, param_grid = param_grid, scoring = 'accuracy', cv = 10, refit = True)
gs.fit(X_train_std, y_train)
print('\nStandardized Logistic Regression grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))
gs.fit(X_train_norm, y_train)
print('\nNormalized Logistic Regression grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))

#Random forest grid search
n_estimators = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]
max_features = ['auto', 'sqrt', 'log2']
param_grid = [{'n_estimators': n_estimators, 'max_features': max_features}]
forest = RandomForestClassifier(random_state = 1)
gs = GridSearchCV(estimator = forest, param_grid = param_grid, scoring = 'accuracy', cv = 10, refit = True)
gs.fit(X_train_std, y_train)
print('\nStandardized Random Forest grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))
gs.fit(X_train_norm, y_train)
print('\nNormalized Random Forest grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))

#KNN grid search
p_neighbors = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
p_weights = ['uniform', 'distance']
p_algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
param_grid = [{'n_neighbors': p_neighbors, 'weights': p_weights, 'algorithm': p_algorithm}]
knn = KNeighborsClassifier()
gs = GridSearchCV(estimator = knn, param_grid = param_grid, scoring = 'accuracy', cv = 10, refit = True)
gs.fit(X_train_std, y_train)
print('\nStandardized KNN grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))
gs.fit(X_train_norm, y_train)
print('\nNormalized KNN grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))

#Perceptron grid search
p_penalty = ['l2', 'l1']
p_alpha = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
p_eta0 = [.001, .01, 1.0]
param_grid = [{'penalty': p_penalty, 'alpha': p_alpha, 'eta0': p_eta0}]
pn = Perceptron(random_state = 1)
gs = GridSearchCV(estimator = pn, param_grid = param_grid, scoring = 'accuracy', cv = 10, refit = True)
gs.fit(X_train_std, y_train)
print('\nStandardized Perceptron grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))
gs.fit(X_train_norm, y_train)
print('\nNormalized Perceptron grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))

#Adaline grid search
param_penalty = ['l2', 'l1']
param_alpha = [.00001, .0001, .001, .01, .1, 1, 10, 100, 1000]
param_eta = [.001, .01, 1.0]
param_grid = [{'penalty': param_penalty, 'alpha': param_alpha, 'eta0': param_eta}]
ada = SGDClassifier(max_iter = 3000, random_state = 1)
gs = GridSearchCV(estimator = ada, param_grid = param_grid, scoring = 'accuracy', cv = 10, refit = True)
gs.fit(X_train_std, y_train)
print('\nStandardized Adaline grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))
gs.fit(X_train_norm, y_train)
print('\nNormalized Adaline grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))

#SVM grid search
param_C = [.0001, .001, .01, 1, 10, 100, 1000, 10000]
#had to remove 'linear' from the kernel parameter because it never converged when it was included
param_kernel = ['rbf','sigmoid','poly']
param_gamma = ['scale','auto']
param_dfs = ['ovo','ovr']
param_grid = [{'C': param_C, 'kernel': param_kernel, 'gamma': param_gamma, 'decision_function_shape': param_dfs}]
svm = SVC(random_state = 1)
gs = GridSearchCV(estimator = svm, param_grid = param_grid, scoring = 'accuracy', cv = 10, refit = True)
gs.fit(X_train_norm, y_train)
print('\nNormalized SVM grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))
gs.fit(X_train_std, y_train)
print('\nStandardized SVM grid search')
print('Best score: %.3f' % (gs.best_score_))
print('Best params: ' + str(gs.best_params_))



#FEATURE IMPORTANCE/SELECTION
#get most important features from random forest using best parameters for all features
#Standardized
feat_labels = df.columns[1:36]
forest = RandomForestClassifier(n_estimators = 200, random_state = 1)
forest.fit(X_train_std, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print('\nRandom Forest standardized feature importance results')
for f in range(X_train_std.shape[1]):
     print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance Standardized')
plt.bar(range(X_train_std.shape[1]), importances[indices], align = 'center')
plt.xticks(range(X_train_std.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train_std.shape[1]])
plt.tight_layout()
plt.show()
#Normalized
feat_labels = df.columns[1:36]
forest = RandomForestClassifier(n_estimators = 160, random_state = 1)
forest.fit(X_train_norm, y_train)
importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]

print('\nRandom Forest normalized feature importance results')
for f in range(X_train_norm.shape[1]):
     print("%2d) %-*s %f" % (f+1, 30, feat_labels[indices[f]], importances[indices[f]]))

plt.title('Feature Importance Normalized')
plt.bar(range(X_train_norm.shape[1]), importances[indices], align = 'center')
plt.xticks(range(X_train_norm.shape[1]), feat_labels[indices], rotation = 90)
plt.xlim([-1, X_train_norm.shape[1]])
plt.tight_layout()
plt.show()


#PCA TESTING 
#The dataset with all features included

print('\n\nAll features PCA testing')

#Logistic Regression PCA testing
lr_std_l = {}
lr_norm_l = {}
lr_std = LogisticRegression(multi_class = 'ovr', C = 1, penalty = 'l1', solver = 'saga', max_iter = 1000, random_state = 1)
lr_std.fit(X_train_std,y_train)
y_pred = lr_std.predict(X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("\nOriginal Standardized Logistic Regression Accuracy: " + str(acc))
lr_norm = LogisticRegression(multi_class = 'ovr', C = 1, penalty = 'l2', solver = 'newton-cg', max_iter = 1000, random_state = 1)
lr_norm.fit(X_train_norm,y_train)
y_pred = lr_norm.predict(X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Logistic Regression Accuracy: " + str(acc))
for i in range (2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    lr_std.fit(X_train_pca, y_train)
    y_pred = lr_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    lr_std_l[i] = acc
max_key = max(lr_std_l, key=lambda k: lr_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Logistic Regression Accuracy: " + str(lr_std_l[max_key]))
for i in range(2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    lr_norm.fit(X_train_pca, y_train)
    y_pred = lr_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    lr_norm_l[i] = acc
max_key = max(lr_norm_l, key=lambda k: lr_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Logistic Regression Accuracy: " + str(lr_norm_l[max_key]))
print("\n")

#Random forest PCA testing
rf_std_l = {}
rf_norm_l = {}
rf_std = RandomForestClassifier(n_estimators = 200, random_state = 1)
rf_std.fit(X_train_std,y_train)
y_pred = rf_std.predict(X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized Random Forest Accuracy: " + str(acc))
rf_norm = RandomForestClassifier(n_estimators = 160, random_state = 1)
rf_norm.fit(X_train_norm,y_train)
y_pred = rf_norm.predict(X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Random Forest Accuracy: " + str(acc))
for i in range (2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    rf_std.fit(X_train_pca, y_train)
    y_pred = rf_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    rf_std_l[i] = acc
max_key = max(rf_std_l, key=lambda k: rf_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Random Forest Accuracy: " + str(rf_std_l[max_key]))
for i in range(2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    rf_norm.fit(X_train_pca, y_train)
    y_pred = rf_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    rf_norm_l[i] = acc
max_key = max(rf_norm_l, key=lambda k: rf_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Random Forests Accuracy: " + str(rf_norm_l[max_key]))
print("\n")

#KNN PCA testing
knn_std_l = {}
knn_norm_l = {}
knn_std = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')
knn_std.fit(X_train_std,y_train)
y_pred = knn_std.predict(X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized KNN Accuracy: " + str(acc))
knn_norm = KNeighborsClassifier(n_neighbors = 12, weights = 'distance')
knn_norm.fit(X_train_norm,y_train)
y_pred = knn_norm.predict(X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized KNN Accuracy: " + str(acc))
for i in range (2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    knn_std.fit(X_train_pca, y_train)
    y_pred = knn_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    knn_std_l[i] = acc
max_key = max(knn_std_l, key=lambda k: knn_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized KNN Accuracy: " + str(knn_std_l[max_key]))
for i in range(2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    knn_norm.fit(X_train_pca, y_train)
    y_pred = knn_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    knn_norm_l[i] = acc
max_key = max(knn_norm_l, key=lambda k: knn_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized KNN Accuracy: " + str(knn_norm_l[max_key]))
print("\n")

#Perceptron PCA testing
pn_std_l = {}
pn_norm_l = {}
pn_std = Perceptron(alpha = 0.00001, eta0 = 0.01, penalty = 'l2', random_state = 1)
pn_std.fit(X_train_std,y_train)
y_pred = pn_std.predict(X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized Perceptron Accuracy: " + str(acc))
pn_norm = Perceptron(alpha = 0.1, eta0 = 0.001, penalty = 'l2', random_state = 1)
pn_norm.fit(X_train_norm,y_train)
y_pred = pn_norm.predict(X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Perceptron Accuracy: " + str(acc))
for i in range (2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    pn_std.fit(X_train_pca, y_train)
    y_pred = pn_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    pn_std_l[i] = acc
max_key = max(pn_std_l, key=lambda k: pn_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Perceptron Accuracy: " + str(pn_std_l[max_key]))
for i in range(2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    pn_norm.fit(X_train_pca, y_train)
    y_pred = pn_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    pn_norm_l[i] = acc
max_key = max(pn_norm_l, key=lambda k: pn_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Perceptron Accuracy: " + str(pn_norm_l[max_key]))
print("\n")

#Adaline PCA testing
ada_std_l = {}
ada_norm_l = {}
ada_std = SGDClassifier(alpha = 1, eta0 = 0.001, penalty = 'l2', max_iter = 3000, random_state = 1)
ada_std.fit(X_train_std,y_train)
y_pred = ada_std.predict(X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized Adaline Accuracy: " + str(acc))
ada_norm = SGDClassifier(alpha = 0.1, eta0 = 0.001, penalty = 'l2', max_iter = 3000, random_state = 1)
ada_norm.fit(X_train_norm,y_train)
y_pred = ada_norm.predict(X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Adaline Accuracy: " + str(acc))
for i in range (2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    ada_std.fit(X_train_pca, y_train)
    y_pred = ada_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    ada_std_l[i] = acc
max_key = max(ada_std_l, key=lambda k: ada_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Adaline Accuracy: " + str(ada_std_l[max_key]))
for i in range(2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    ada_norm.fit(X_train_pca, y_train)
    y_pred = ada_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    ada_norm_l[i] = acc
max_key = max(ada_norm_l, key=lambda k: ada_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Adaline Accuracy: " + str(ada_norm_l[max_key]))
print("\n")

#SVM PCA testing
svm_std_l = {}
svm_norm_l = {}
svm_std = SVC(C = 1, decision_function_shape = 'ovo', gamma = 'scale', kernel = 'sigmoid', random_state = 1)
svm_std.fit(X_train_std,y_train)
y_pred = svm_std.predict(X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized SVM Accuracy: " + str(acc))
svm_norm = SVC(C = 1, decision_function_shape = 'ovo', gamma = 'scale', kernel = 'rbf', random_state = 1)
svm_norm.fit(X_train_norm,y_train)
y_pred = svm_norm.predict(X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized SVM Accuracy: " + str(acc))
for i in range (2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    svm_std.fit(X_train_pca, y_train)
    y_pred = svm_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    svm_std_l[i] = acc
max_key = max(svm_std_l, key=lambda k: svm_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized SVM Accuracy: " + str(svm_std_l[max_key]))
for i in range(2,35):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    svm_norm.fit(X_train_pca, y_train)
    y_pred = svm_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    svm_norm_l[i] = acc
max_key = max(svm_norm_l, key=lambda k: svm_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized SVM Accuracy: " + str(svm_norm_l[max_key]))
print("\n")


#this is the dataset with the unimportant features removed
#Removed Features: gender, Bravery, Hardwork, Intelligence, Ambition, Daring, Dedication, Knowledge, Cunning, Extraverted, Agreeable, Clever
lil_X_train = np.array(train.iloc[:, [1,2,3,5,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]])
lil_X_develop = np.array(validate.iloc[:, [1,2,3,5,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]])
lil_X_test = np.array(test.iloc[:, [1,2,3,5,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35]])

#standardize data
sc = StandardScaler()
sc.fit(lil_X_train)
lil_X_train_std = sc.transform(lil_X_train)
lil_X_develop_std = sc.transform(lil_X_develop)
lil_X_test_std = sc.transform(lil_X_test)

#normalize data
mms = MinMaxScaler()
lil_X_train_norm = mms.fit_transform(lil_X_train)
lil_X_develop_norm = mms.fit_transform(lil_X_develop)
lil_X_test_norm = mms.transform(lil_X_test)


#Now rerun the PCA testing but with the unimportant features removed
print('PCA testing with Removed Features')

#Logistic Regression PCA testing
lr_std_l = {}
lr_norm_l = {}
lr_std = LogisticRegression(multi_class = 'ovr', C = 1, penalty = 'l1', solver = 'saga', max_iter = 1000, random_state = 1)
lr_std.fit(lil_X_train_std,y_train)
y_pred = lr_std.predict(lil_X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("\nOriginal Standardized Logistic Regression Accuracy: " + str(acc))
lr_norm = LogisticRegression(multi_class = 'ovr', C = 1, penalty = 'l2', solver = 'newton-cg', max_iter = 1000, random_state = 1)
lr_norm.fit(lil_X_train_norm,y_train)
y_pred = lr_norm.predict(lil_X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Logistic Regression Accuracy: " + str(acc))
for i in range (2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    lr_std.fit(X_train_pca, y_train)
    y_pred = lr_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    lr_std_l[i] = acc
max_key = max(lr_std_l, key=lambda k: lr_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Logistic Regression Accuracy: " + str(lr_std_l[max_key]))
for i in range(2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    lr_norm.fit(X_train_pca, y_train)
    y_pred = lr_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    lr_norm_l[i] = acc
max_key = max(lr_norm_l, key=lambda k: lr_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Logistic Regression Accuracy: " + str(lr_norm_l[max_key]))
print("\n")

#Random forest PCA testing
rf_std_l = {}
rf_norm_l = {}
rf_std = RandomForestClassifier(n_estimators = 200, random_state = 1)
rf_std.fit(lil_X_train_std,y_train)
y_pred = rf_std.predict(lil_X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized Random Forest Accuracy: " + str(acc))
rf_norm = RandomForestClassifier(n_estimators = 160, random_state = 1)
rf_norm.fit(lil_X_train_norm,y_train)
y_pred = rf_norm.predict(lil_X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Random Forest Accuracy: " + str(acc))
for i in range (2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    rf_std.fit(X_train_pca, y_train)
    y_pred = rf_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    rf_std_l[i] = acc
max_key = max(rf_std_l, key=lambda k: rf_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Random Forest Accuracy: " + str(rf_std_l[max_key]))
for i in range(2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    rf_norm.fit(X_train_pca, y_train)
    y_pred = rf_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    rf_norm_l[i] = acc
max_key = max(rf_norm_l, key=lambda k: rf_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Random Forests Accuracy: " + str(rf_norm_l[max_key]))
print("\n")

#KNN PCA testing
knn_std_l = {}
knn_norm_l = {}
knn_std = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')
knn_std.fit(lil_X_train_std,y_train)
y_pred = knn_std.predict(lil_X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized KNN Accuracy: " + str(acc))
knn_norm = KNeighborsClassifier(n_neighbors = 12, weights = 'distance')
knn_norm.fit(lil_X_train_norm,y_train)
y_pred = knn_norm.predict(lil_X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized KNN Accuracy: " + str(acc))
for i in range (2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    knn_std.fit(X_train_pca, y_train)
    y_pred = knn_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    knn_std_l[i] = acc
max_key = max(knn_std_l, key=lambda k: knn_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized KNN Accuracy: " + str(knn_std_l[max_key]))
for i in range(2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    knn_norm.fit(X_train_pca, y_train)
    y_pred = knn_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    knn_norm_l[i] = acc
max_key = max(knn_norm_l, key=lambda k: knn_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized KNN Accuracy: " + str(knn_norm_l[max_key]))
print("\n")

#Perceptron PCA testing
pn_std_l = {}
pn_norm_l = {}
pn_std = Perceptron(alpha = 0.00001, eta0 = 0.01, penalty = 'l2', random_state = 1)
pn_std.fit(lil_X_train_std,y_train)
y_pred = pn_std.predict(lil_X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized Perceptron Accuracy: " + str(acc))
pn_norm = Perceptron(alpha = 0.1, eta0 = 0.001, penalty = 'l2', random_state = 1)
pn_norm.fit(lil_X_train_norm,y_train)
y_pred = pn_norm.predict(lil_X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Perceptron Accuracy: " + str(acc))
for i in range (2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_std)
    X_develop_pca = pca.transform(X_develop_std)
    pn_std.fit(X_train_pca, y_train)
    y_pred = pn_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    pn_std_l[i] = acc
max_key = max(pn_std_l, key=lambda k: pn_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Perceptron Accuracy: " + str(pn_std_l[max_key]))
for i in range(2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(X_train_norm)
    X_develop_pca = pca.transform(X_develop_norm)
    pn_norm.fit(X_train_pca, y_train)
    y_pred = pn_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    pn_norm_l[i] = acc
max_key = max(pn_norm_l, key=lambda k: pn_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Perceptron Accuracy: " + str(pn_norm_l[max_key]))
print("\n")

#Adaline PCA testing
ada_std_l = {}
ada_norm_l = {}
ada_std = SGDClassifier(alpha = 1, eta0 = 0.001, penalty = 'l2', max_iter = 3000, random_state = 1)
ada_std.fit(lil_X_train_std,y_train)
y_pred = ada_std.predict(lil_X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized Adaline Accuracy: " + str(acc))
ada_norm = SGDClassifier(alpha = 0.1, eta0 = 0.001, penalty = 'l2', max_iter = 3000, random_state = 1)
ada_norm.fit(lil_X_train_norm,y_train)
y_pred = ada_norm.predict(lil_X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized Adaline Accuracy: " + str(acc))
for i in range (2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(lil_X_train_std)
    X_develop_pca = pca.transform(lil_X_develop_std)
    ada_std.fit(X_train_pca, y_train)
    y_pred = ada_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    ada_std_l[i] = acc
max_key = max(ada_std_l, key=lambda k: ada_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized Adaline Accuracy: " + str(ada_std_l[max_key]))
for i in range(2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(lil_X_train_norm)
    X_develop_pca = pca.transform(lil_X_develop_norm)
    ada_norm.fit(X_train_pca, y_train)
    y_pred = ada_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    ada_norm_l[i] = acc
max_key = max(ada_norm_l, key=lambda k: ada_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized Adaline Accuracy: " + str(ada_norm_l[max_key]))
print("\n")

#SVM PCA testing
svm_std_l = {}
svm_norm_l = {}
svm_std = SVC(C = 1, decision_function_shape = 'ovo', gamma = 'scale', kernel = 'sigmoid', random_state = 1)
svm_std.fit(lil_X_train_std,y_train)
y_pred = svm_std.predict(lil_X_develop_std)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Standardized SVM Accuracy: " + str(acc))
svm_norm = SVC(C = 1, decision_function_shape = 'ovo', gamma = 'scale', kernel = 'rbf', random_state = 1)
svm_norm.fit(lil_X_train_norm,y_train)
y_pred = svm_norm.predict(lil_X_develop_norm)
acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
print("Original Normalized SVM Accuracy: " + str(acc))
for i in range (2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(lil_X_train_std)
    X_develop_pca = pca.transform(lil_X_develop_std)
    svm_std.fit(X_train_pca, y_train)
    y_pred = svm_std.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    svm_std_l[i] = acc
max_key = max(svm_std_l, key=lambda k: svm_std_l[k])
print("Best PCA: " + str(max_key) + " Standardized SVM Accuracy: " + str(svm_std_l[max_key]))
for i in range(2,21):
    pca = PCA(n_components = i)
    X_train_pca = pca.fit_transform(lil_X_train_norm)
    X_develop_pca = pca.transform(lil_X_develop_norm)
    svm_norm.fit(X_train_pca, y_train)
    y_pred = svm_norm.predict(X_develop_pca)
    acc = accuracy_score(y_true = y_develop, y_pred = y_pred)
    svm_norm_l[i] = acc
max_key = max(svm_norm_l, key=lambda k: svm_norm_l[k])
print("Best PCA: " + str(max_key) + " Normalized SVM Accuracy: " + str(svm_norm_l[max_key]))
print("\n")


##FINAL TESTING
#This runs each model with their best hyperparameters and predicts on the TEST data
#Each model also uses either all features or the data with removed features depending on which performed best
#We used both accuracy and F1 scores to score our models' performances

print('Final test results')

#Logistic Regression Final Test
lr_std = LogisticRegression(multi_class = 'ovr', C = 1, penalty = 'l1', solver = 'saga', max_iter = 1000, random_state = 1)
lr_norm = LogisticRegression(multi_class = 'ovr', C = 1, penalty = 'l2', solver = 'newton-cg', max_iter = 1000, random_state = 1)
#PCA
pca_std = PCA(n_components = 24)
pca_norm = PCA(n_components = 13)
X_train_pca_std = pca_std.fit_transform(X_train_std)
X_test_pca_std = pca_std.transform(X_test_std)
X_train_pca_norm = pca_norm.fit_transform(lil_X_train_norm)
X_test_pca_norm = pca_norm.transform(lil_X_test_norm)
#Normalized
lr_norm.fit(X_train_pca_norm, y_train)
y_pred = lr_norm.predict(X_test_pca_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('\nNormalized Logistic Regression Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Normalized Logistic Regression: %.3f' % (f1))
#Standardized
lr_std.fit(X_train_pca_std, y_train)
y_pred = lr_std.predict(X_test_pca_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('Standardized Logistic Regression Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Standardized Logistic Regression: %.3f' % (f1))

#Random Forest Final Test
rf_std = RandomForestClassifier(n_estimators = 200, random_state = 1)
rf_norm = RandomForestClassifier(n_estimators = 160, random_state = 1)
#Normalized
rf_norm.fit(lil_X_train_norm, y_train)
y_pred = rf_norm.predict(lil_X_test_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('\nNormalized Random Forest Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Normalized Random Forest: %.3f' % (f1))
#Standardized
rf_std.fit(lil_X_train_std, y_train)
y_pred = rf_std.predict(lil_X_test_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('Standardized Random Forest Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Standardized Random Forest: %.3f' % (f1))

#KNN Final Test
knn_std = KNeighborsClassifier(n_neighbors = 13, weights = 'uniform')
knn_norm = KNeighborsClassifier(n_neighbors = 12, weights = 'distance')
#PCA
pca_std = PCA(n_components = 17)
pca_norm = PCA(n_components = 4)
X_train_pca_std = pca_std.fit_transform(lil_X_train_std)
X_test_pca_std = pca_std.transform(lil_X_test_std)
X_train_pca_norm = pca_norm.fit_transform(lil_X_train_norm)
X_test_pca_norm = pca_norm.transform(lil_X_test_norm)
#Normalized
knn_norm.fit(X_train_pca_norm, y_train)
y_pred = knn_norm.predict(X_test_pca_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('\nNormalized KNN Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Normalized KNN: %.3f' % (f1))
#Standardized
knn_std.fit(X_train_pca_std, y_train)
y_pred = knn_std.predict(X_test_pca_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('Standardized KNN Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Standardized KNN: %.3f' % (f1))

#Perceptron Final Test
pn_std = Perceptron(alpha = 0.00001, eta0 = 0.01, penalty = 'l2', random_state = 1)
pn_norm = Perceptron(alpha = 0.01, eta0 = 0.001, penalty = 'l2', random_state = 1)
#PCA
pca_std = PCA(n_components = 17)
pca_norm = PCA(n_components = 15)
X_train_pca_std = pca_std.fit_transform(X_train_std)
X_test_pca_std = pca_std.transform(X_test_std)
X_train_pca_norm = pca_norm.fit_transform(lil_X_train_norm)
X_test_pca_norm = pca_norm.transform(lil_X_test_norm)
#Normalized
pn_norm.fit(X_train_pca_norm, y_train)
y_pred = pn_norm.predict(X_test_pca_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('\nNormalized Perceptron with PCA Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Normalized Perceptron with PCA: %.3f' % (f1))
#Standardization
pn_std.fit(X_train_pca_std, y_train)
y_pred = pn_std.predict(X_test_pca_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('Standardized Perceptron with PCA Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Standardized Perceptron with PCA: %.3f' % (f1))

#Adaline Final Test
ada_std = SGDClassifier(alpha = 1, eta0 = 0.001, penalty = 'l2', max_iter = 3000, random_state = 1)
ada_norm = SGDClassifier(alpha = 0.1, eta0 = 0.001, penalty = 'l2', max_iter = 3000, random_state = 1)
#PCA
pca_std = PCA(n_components = 18)
pca_norm = PCA(n_components = 15)
X_train_pca_std = pca_std.fit_transform(lil_X_train_std)
X_test_pca_std = pca_std.transform(lil_X_test_std)
X_train_pca_norm = pca_norm.fit_transform(lil_X_train_norm)
X_test_pca_norm = pca_norm.transform(lil_X_test_norm)
#Normalized
ada_norm.fit(X_train_pca_norm, y_train)
y_pred = ada_norm.predict(X_test_pca_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('\nNormalized Adaline with PCA Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Normalized Adaline with PCA: %.3f' % (f1))
#Standardization
ada_std.fit(X_train_pca_std, y_train)
y_pred = ada_std.predict(X_test_pca_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('Standardized Adaline with PCA Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Standardized Adaline with PCA: %.3f' % (f1))

#SVM Final Test
svm_std = SVC(C = 1, kernel = 'sigmoid', decision_function_shape = 'ovo', gamma = 'scale')
svm_norm = SVC(C = 1, kernel = 'rbf', decision_function_shape = 'ovo', gamma = 'scale')
#PCA
pca_std = PCA(n_components = 33)
pca_norm = PCA(n_components = 4)
X_train_pca_std = pca_std.fit_transform(X_train_std)
X_test_pca_std = pca_std.transform(X_test_std)
X_train_pca_norm = pca_norm.fit_transform(lil_X_train_norm)
X_test_pca_norm = pca_norm.transform(lil_X_test_norm)
#Normalized
svm_norm.fit(X_train_pca_norm, y_train)
y_pred = svm_norm.predict(X_test_pca_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('\nNormalized SVM Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Normalized SVM: %.3f' % (f1))
#Standardized
svm_std.fit(X_train_pca_std, y_train)
y_pred = svm_std.predict(X_test_pca_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print('Standardized SVM Accuracy: %.3f' % (acc))
f1 = f1_score(y_test, y_pred, average = 'macro')
print('F1 score for Standardized SVM: %.3f' % (f1))

#DUMMY MODEL
#Normalized
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train_norm, y_train)
y_pred = dummy_clf.predict(X_test_norm)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print("\nDummy Model Normalized Accuracy: " + str(acc))
f1 = f1_score(y_test,y_pred,average = "macro")
print("Dummy Model Normalized F1 Score: " + str(f1))
#Standardized
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(X_train_std, y_train)
y_pred = dummy_clf.predict(X_test_std)
acc = accuracy_score(y_true = y_test, y_pred = y_pred)
print("Dummy Model Standardized Accuracy: " + str(acc))
f1 = f1_score(y_test,y_pred,average = "macro")
print("Dummy Model Standardized F1 Score: " + str(f1))
