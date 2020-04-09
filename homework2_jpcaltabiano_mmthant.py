import numpy as np
from matplotlib import pyplot as plt

DEBUG = True

# Given an array of faces (N x M x M, where N is number of examples and M is number of pixes along each axis),
# return a design matrix Xtilde ((M**2 + 1) x N) whose last row contains all 1s.
def reshapeAndAppend1s (faces):
    faces = faces[ :, :, ::-1]
    faces = faces.T
    faces = np.reshape(faces, (faces.shape[0] ** 2, faces.shape[2]))
    faces = np.vstack((faces, np.ones(faces.shape[1])))
    return faces 

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, return the (unregularized)
# MSE.
def fMSE (w, Xtilde, y):
    yhat = Xtilde.T.dot(w) 
    fmse = ((yhat-y)**2).mean() / 2
    return fmse

# Given a vector of weights w, a design matrix Xtilde, and a vector of labels y, and a regularization strength
# alpha (default value of 0), return the gradient of the (regularized) MSE loss.
def gradfMSE (w, Xtilde, y, alpha = 0.):
    if alpha == 0.0: 
        regularization = 0
    else:
        regularization = (w[:-1].T.dot(w[:-1])).mean() * (alpha/2)
    gradfmse = (Xtilde * (Xtilde.T.dot(w) - y)).mean() + regularization
    return gradfmse
    

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using the analytical solution.
def method1 (Xtilde, y):
    A = Xtilde.dot(Xtilde.T)
    B = Xtilde.dot(y)
    w = np.linalg.solve(A, B)
    return w

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE.
def method2 (Xtilde, y):
    return gradientDescent(Xtilde, y)

# Given a design matrix Xtilde and labels y, train a linear regressor for Xtilde and y using gradient descent on fMSE
# with regularization.
def method3 (Xtilde, y):
    ALPHA = 0.1
    return gradientDescent(Xtilde, y, alpha = ALPHA)

# Helper method for method2 and method3.
def gradientDescent (Xtilde, y, alpha = 0.):
    EPSILON = 3e-3  # Step size aka learning rate
    T = 5000  # Number of gradient descent iterations
    w = 0.01 * np.random.randn(Xtilde.shape[0])
    for i in range(T):
        if i % 500 == 0: 
            print(f'epoch {i}')
        w = w - (EPSILON * gradfMSE(w, Xtilde, y, alpha))
    return w

if __name__ == "__main__":
    # Load data
    Xtilde_tr = reshapeAndAppend1s(np.load("age_regression_Xtr.npy"))
    ytr = np.load("age_regression_ytr.npy")
    Xtilde_te = reshapeAndAppend1s(np.load("age_regression_Xte.npy"))
    yte = np.load("age_regression_yte.npy")

    w1 = method1(Xtilde_tr, ytr)
    train_loss_w1 = fMSE(w1, Xtilde_tr, ytr)
    test_loss_w1 = fMSE(w1, Xtilde_te, yte)
    
    w2 = method2(Xtilde_tr, ytr)
    train_loss_w2 = fMSE(w2, Xtilde_tr, ytr)
    test_loss_w2 = fMSE(w2, Xtilde_te, yte)

    w3 = method3(Xtilde_tr, ytr)
    train_loss_w3 = fMSE(w3, Xtilde_tr, ytr)
    test_loss_w3 = fMSE(w3, Xtilde_te, yte)
    # Report fMSE cost using each of the three learned weight vectors
    # ...b 

    # report rmse for part c
    yhat = Xtilde_te.T.dot(w3)
    rmse = (((yhat-yte)**2).mean()) ** 0.5

    errors = abs(yte - yhat)
    errors_idx = np.argsort(errors)
    errors_idx = errors_idx[::-1]
    errors_idx = errors_idx[0:5]


    if DEBUG:
        # print("Weight Vector for method 1:\t", w1)
        print("Train Loss for method 1:\t", train_loss_w1)
        print("Test Loss for method 1:\t", test_loss_w1 , "\n")
        # print("Weight Vector for method 2:\t", w3)
        print("Train Loss for method 2:\t", train_loss_w2)
        print("Test Loss for method 2:\t", test_loss_w2, "\n")
        # print("Weight Vector for method 3:\t", w3)
        print("Train Loss for method 3:\t", train_loss_w3)
        print("Test Loss for method 3:\t", test_loss_w3, "\n")

    for i in [w1, w2, w3]:
        t = i[:2304].reshape((48,48))
        plt.imshow(t)
        plt.show()

    for i in errors_idx:
        print(i)
        image = Xtilde_te[:-1, i].reshape((48,48))
        plt.imshow(image)
        plt.show()
        


