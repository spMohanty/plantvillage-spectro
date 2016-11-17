#!/usr/bin/env python
import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA


"""
Read data
"""
A = np.asarray(pd.read_csv("data/KBH_A_0901.csv"))
A_1 = np.asarray(pd.read_csv("data/KBH_A_0902.csv"))

A = np.concatenate([A_1, A], axis=0)
H = np.asarray(pd.read_csv("data/KBH_H_0901.csv"))
S = np.asarray(pd.read_csv("data/KBH_S_0903.csv"))

A_label = np.zeros(A.shape[0]) + 0
H_label = np.zeros(H.shape[0]) + 1
S_label = np.zeros(S.shape[0]) + 2

X_spectro = np.concatenate([A, H, S], axis=0)
Y_spectro = np.concatenate([A_label, H_label, S_label], axis=0)

"""
Implement Classification Algorithms
"""

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifiers = [
    KNeighborsClassifier(1),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0), warm_start=True),
    DecisionTreeClassifier(max_depth=20),
    RandomForestClassifier(max_depth=20, n_estimators=20, max_features=50),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
    ]

datasets = [(X_spectro, Y_spectro)]

figure = plt.figure(figsize=(27, 9))

for test_ratio in [0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    print "="*100
    print "Test Ratio : ", test_ratio
    print
    i = 1
    output = ""
    # iterate over datasets
    for ds_cnt, ds in enumerate(datasets):
        # preprocess dataset, split into training and test part
        X, y = ds
        X = StandardScaler().fit_transform(X)

        pca = PCA(n_components=2)
        pca.fit(X)
        X_projected = pca.transform(X)

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=test_ratio, random_state=42)


        X_train_projected = pca.transform(X_train)
        X_test_projected = pca.transform(X_test)

        x_min, x_max = X_projected[:, 0].min() - .5, X_projected[:, 0].max() + .5
        y_min, y_max = X_projected[:, 1].min() - .5, X_projected[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title("Input data")
        # Plot the training points
        ax.scatter(X_train_projected[:, 0], X_train_projected[:, 1], c=y_train, cmap=cm_bright)
        # and testing points
        ax.scatter(X_test_projected[:, 0], X_test_projected[:, 1], c=y_test, cmap=cm_bright, alpha=0.6)
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1
        #
        # iterate over classifiers
        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
            # output += "="*100 +"\n"
            # print "="*100 +"\n"
            output += "Algorithm : \"" + name
            print "Algorithm : \"" + name,
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            output += "\" Accuracy : \"" + str(score*100) + "%\"" +"\n"
            print "\" Accuracy : \"", score*100, "%\""


            f = open("outputs/"+str(test_ratio)+".txt", "w")
            f.write(output)
            f.close()
            # # Plot the decision boundary. For that, we will assign a color to each
            # # point in the mesh [x_min, x_max]x[y_min, y_max].

            # print np.c_[xx.ravel(), yy.ravel()].shape


            # if hasattr(clf, "decision_function"):
            #     Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            # else:
            #     Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
            # #
            # # Put the result into a color plot
            # Z = Z.reshape(xx.shape)
            # ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            #
            # # Plot also the training points
            # ax.scatter(X_train_projected[:, 0], X_train_projected[:, 1], c=y_train, cmap=cm_bright)
            # # and testing points
            # ax.scatter(X_test_projected[:, 0], X_test_projected[:, 1], c=y_test, cmap=cm_bright,
            #            alpha=0.6)
            #
            # ax.set_xlim(xx.min(), xx.max())
            # ax.set_ylim(yy.min(), yy.max())
            # ax.set_xticks(())
            # ax.set_yticks(())
            # if ds_cnt == 0:
            #     ax.set_title(name)
            # ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score*100).lstrip('0'),
            #         size=15, horizontalalignment='right')
            i += 1


plt.tight_layout()
plt.savefig("classifier_results.png")
