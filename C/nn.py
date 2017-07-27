#!/tool/pandora64/bin/python3

import numpy as np

number_of_iterations = 1
reg = 1e-3
step_size =  1

N = 3
D = 2
K = 3
h = 2
X = np.zeros((N*K,D))
y = np.zeros(N*K,dtype='uint8')
for j in range(K):
    ix = range(N*j,(N*(j+1)))
    r = np.linspace(0.0,1,N)
    t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2
    X[ix] = np.c_[r*np.sin(t),r*np.cos(t)]
    y[ix] = j

W = 0.01*np.random.randn(D,h);
b = np.zeros((1,h))
W2 = 0.01*np.random.randn(h,K);
b2 = np.zeros((1,K))

print("X=")
print(X)
print("y=")
print(y)
print("W initialized as:")
print(W)
print("b initialized as:")
print(b)
print("W2 initialized as:")
print(W2)
print("b2 initialized as:")
print(b2)


data_size = X.shape[0];
for iteration in range(number_of_iterations):
    # forward propagation
    hidden_before_relu = np.dot(X,W) + b
    print("Before ReLU, H = ")
    print(hidden_before_relu)
    hidden = np.maximum(0,hidden_before_relu)
    print("After ReLU, H = ")
    print(hidden)
    score = np.dot(hidden,W2) + b2
    print("score = ")
    print(score)
    score_exp = np.exp(score)
    probs = score_exp / np.sum(score_exp,axis=1,keepdims=True)
    correct_loss = probs[range(data_size),y]
    logged_loss = -np.log(correct_loss)
    data_loss = np.sum(logged_loss) / data_size
    print("data loss = %f" % data_loss)
    reg_loss = 0.5*reg*np.sum(W*W)+0.5*reg*np.sum(W2*W2)
    print("reg loss = %f" % reg_loss)
    loss = data_loss + reg_loss
    print("Iteration %d: loss = %f, data_loss = %f, reg_loss = %f" % (iteration, loss, data_loss, reg_loss))

    # backward propagation
    dscore = probs
    dscore[range(data_size),y] -= 1
    dscore /= data_size
    print("dscore = ")
    print(dscore)
    dW2 = np.dot(hidden.T,dscore)
    print("dW2 = ")
    print(dW2)
    db2 = np.sum(dscore,axis=0,keepdims=True)
    print("db2 = ")
    print(db2)
    dhidden = np.dot(dscore,W2.T)
    print("dH before ReLU")
    print(dhidden)
    dhidden[hidden <= 0] = 0
    print("dH after ReLU")
    print(dhidden)
    dW = np.dot(X.T,dhidden)
    print("dW = ")
    print(dW)
    db = np.sum(dhidden,axis=0,keepdims=True)
    print("db = ")
    print(db)
    # update weights
    dW += reg*W
    dW2 += reg*W2
    print("After Reg backprop, dW2 = ")
    print(dW2)
    print("After Reg backprop, dW = ")
    print(dW)
    b += -step_size*db
    W += -step_size*dW
    b2 += -step_size*db2
    W2 += -step_size*dW2
    print("After update")
    print("W = ")
    print(W)
    print("b = ")
    print(b)
    print("W2 = ")
    print(W2)
    print("b2 = ")
    print(b2)


hidden = np.maximum(0,np.dot(X,W) + b)
test_scores = np.dot(hidden, W2) + b2
predicted = np.argmax(test_scores, axis=1)
print("Correctness: %f" % (np.mean(predicted == y)))


