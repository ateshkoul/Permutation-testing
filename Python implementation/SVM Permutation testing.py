# -*- coding: utf-8 -*-
# Author Atesh Koul
print(__doc__)

import numpy as np

import matplotlib.pyplot as plt
import scipy.io

from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold, permutation_test_score


mat = scipy.io.loadmat('E:\Research_Project\Kinematics\Action_execution\Permutation_testing\Data\Data_Z_removed_random_value_mean_sd_filled_all_NaN_Andrea_method.mat')
data = mat['fullData']

#python indexing starts at 0, so the column starts at 1 instead of 2
initial_col = np.array([1,11,21,31,41,51,61,71,81,91,101,111])
y = data[:,0]
for x in range(0, 10):
    columns = initial_col+x
    X = data[:,columns]
    svm = SVC(kernel='linear')
    cv = StratifiedKFold(y, 2)

    score, permutation_scores, pvalue = permutation_test_score(
        svm, X, y, scoring="accuracy", cv=cv, n_permutations=10, n_jobs=1)

    print("Classification score %s (pvalue : %s)" % (score, pvalue))