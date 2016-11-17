#!/usr/bin/env python

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import PCA

from sklearn.metrics import mean_squared_error

np.set_printoptions(suppress=True)
print


A = np.asarray(pd.read_csv("data/KBH_A_0901.csv"))
A_1 = np.asarray(pd.read_csv("data/KBH_A_0902.csv"))

A = np.concatenate([A_1, A], axis=0)
H = np.asarray(pd.read_csv("data/KBH_H_0901.csv"))
S = np.asarray(pd.read_csv("data/KBH_S_0903.csv"))

#Compute mean distance
#AH
_sum = 0
_count = 0
for _i in range(A.shape[0]):
    for _j in range(H.shape[0]):
        _sum += mean_squared_error([A[_i]], [H[_j]])
        _count += 1

print "Mean Distance AH: ", _sum/_count
#HS
_sum = 0
_count = 0
for _i in range(H.shape[0]):
    for _j in range(S.shape[0]):
        _sum += mean_squared_error([H[_i]], [S[_j]])
        _count += 1

print "Mean Distance HS: ", _sum/_count
#AS

_sum = 0
_count = 0
for _i in range(A.shape[0]):
    for _j in range(S.shape[0]):
        _sum += mean_squared_error([A[_i]], [S[_j]])
        _count += 1

print "Mean Distance AS: ", _sum/_count

# model = TSNE(n_components=2, random_state=0)
model = PCA(n_components=2)

projections = model.fit_transform(np.concatenate([A, H, S]))

A_ = projections[:len(A)]
H_ = projections[len(A):len(A)+len(H)]
S_ = projections[len(A)+len(H):]

plt.figure(2, figsize=(8, 6))
fig = plt.figure(1, figsize=(8, 6))

# ax = Axes3D(fig)
# ax.scatter(A_[:,0], A_[:, 1], A_[:, 2],  color='g', alpha=0.5)
# ax.scatter(H_[:,0], H_[:, 1], H_[:, 2],  color='r', alpha=0.5)
# ax.scatter(S_[:,0], S_[:, 1], S_[:, 2],  color='b', alpha=0.5)

plt.scatter(A_[:,0], A_[:, 1],  color='g', marker="o", alpha=0.7) #asymptomatic
plt.scatter(H_[:,0], H_[:, 1],  color='r', marker="+", alpha=0.7) #healthy
plt.scatter(S_[:,0], S_[:, 1], color='b', marker="x", alpha=0.7) #symptomatics



# plt.scatter(A_[:, 0].tolist(), A_[:, 1].tolist(), '.r')
# plt.scatter(H_[:, 0], H_[:, 1], 'b')
# plt.scatter(S_[:, 0], S_[:, 1], 'g')

plt.savefig("tsne.png")
