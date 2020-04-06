import numpy as np
import homework2_jpcaltabiano_mmthant as hw2

def test_reshapeAndAppend1s():
    faces = np.load("age_regression_Xtr.npy")
    X_tilde_tr = hw2.reshapeAndAppend1s(faces)
    print(X_tilde_tr.shape)
    

test_reshapeAndAppend1s()
