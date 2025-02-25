import numpy as np

def softmax(z):
    assert len(z.shape) == 2

    s = np.max(z, axis=1)
    s = s[:, np.newaxis]
    e_x = np.exp(z-s)
    div = np.sum(e_x, axis=1)
    div = div[:,np.newaxis]
    return e_x/div

x1 = np.array([[1.3, 5.1, 2.2, 0.7, 1.1]])
print(softmax(x1))