#!/usr/bin/env python
import pandas as pd
import numpy as np


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

X = np.concatenate([A, H, S], axis=0)
Y = np.concatenate([A_label, H_label, S_label], axis=0)

print X.shape
print Y.shape
