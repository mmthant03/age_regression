import numpy as np
import homework2_jpcaltabiano_mmthant as hw2

Xtr = np.load("age_regression_Xtr.npy")
ytr = np.load("age_regression_ytr.npy")
Xte = np.load("age_regression_Xte.npy")
yte = np.load("age_regression_yte.npy")

def test_reshapeAndAppend1s():
    X_tilde = hw2.reshapeAndAppend1s(Xtr)
    print(X_tilde.shape)
    
def test_fMSE():
    X_tilde = hw2.reshapeAndAppend1s(Xtr)


test_reshapeAndAppend1s()
