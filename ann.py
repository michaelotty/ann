import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)

    return 1/(1+np.exp(-x))

# input dataset
X = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1] ])

# output dataset
y = np.array([[0],
              [1],
              [1],
              [0]])

# seed random numbers to make calculation
# deterministic 
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1

for j in range(600000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss the target value?
    l2_error = y - l2

    if (j% 10000) == 0:
        print("Error:" + str(np.mean(np.abs(l2_error))))

    # in what direction is the target value
    l2_delta = l2_error * nonlin(l2,deriv=True)

    # how much did we miss?
    l1_error = l2_delta.dot(syn1.T)

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,deriv=True)

    # update weights
    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)

