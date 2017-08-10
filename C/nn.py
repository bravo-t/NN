#!/tool/pandora64/bin/python3

import numpy as np

def batchnorm_forward(x, gamma, beta, mode,running_mean,running_var):
    eps = 1e-5
    momentum = 0.0
    N, D = x.shape

    out, cache = None, None
    if mode == 'train':    
        sample_mean = np.mean(x, axis=0, keepdims=True)       # [1,D]    
        sample_var = np.var(x, axis=0, keepdims=True)         # [1,D] 
        x_normalized = (x - sample_mean) / np.sqrt(sample_var + eps)    # [N,D]    
        out = gamma * x_normalized + beta    
        cache = (x_normalized, gamma, beta, sample_mean, sample_var, x, eps)    
        running_mean = momentum * running_mean + (1 - momentum) * sample_mean    
        running_var = momentum * running_var + (1 - momentum) * sample_var
        return out, x_normalized, sample_mean, sample_var,running_mean,running_var
    elif mode == 'test':    
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)    
        out = gamma * x_normalized + beta
        return out, None
    else:    
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)


    return out, x_normalized, sample_mean, sample_var,running_mean,running_var

def batchnorm_backward(dout, x_normalized, gamma, beta, sample_mean, sample_var, x, eps):
    dx, dgamma, dbeta = None, None, None
    #x_normalized, gamma, beta, sample_mean, sample_var, x, eps = cache
    N, D = x.shape
    dx_normalized = dout * gamma       # [N,D]
    x_mu = x - sample_mean             # [N,D]
    sample_std_inv = 1.0 / np.sqrt(sample_var + eps)    # [1,D]
    dsample_var = -0.5 * np.sum(dx_normalized * x_mu, axis=0, keepdims=True) * sample_std_inv**3
    dsample_mean = -1.0 * np.sum(dx_normalized * sample_std_inv, axis=0, keepdims=True) - 2.0 * dsample_var * np.mean(x_mu, axis=0, keepdims=True)
    dx1 = dx_normalized * sample_std_inv
    dx2 = 2.0/N * dsample_var * x_mu
    dx = dx1 + dx2 + 1.0/N * dsample_mean
    dgamma = np.sum(dout * x_normalized, axis=0, keepdims=True)
    dbeta = np.sum(dout, axis=0, keepdims=True)

    return dx, dgamma, dbeta

number_of_iterations = 2
reg = 1e-3
step_size =  1

N = 3
D = 2
K = 3
h = 4
mu=0.5
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
vW = np.zeros((D,h))
vb = np.zeros((1,h))
vW2 = np.zeros((h,K))
vb2 = np.zeros((1,K))

gamma = np.ones((1,h))
beta = np.zeros((1,h))
running_mean = np.zeros((1,h))
running_var = np.zeros((1,h))

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

print("vW initialized as:")
print(vW)
print("vb initialized as:")
print(vb)
print("vW2 initialized as:")
print(vW2)
print("vb2 initialized as:")
print(vb2)


data_size = X.shape[0];
for iteration in range(number_of_iterations):
    # forward propagation
    hidden_before_relu = np.dot(X,W) + b
    print("Before ReLU, H = ")
    print(hidden_before_relu)
    print("Before batchnorm_forward, H = ")
    print(hidden_before_relu)
    print("Before batchnorm_forward, gamma = ")
    print(gamma)
    print("Before batchnorm_forward, beta = ")
    print(beta)
    print("Before batchnorm_forward, running_mean = ")
    print(running_mean)
    print("Before batchnorm_forward, running_var = ")
    print(running_var)
    hidden_before_relu, x_normalized, sample_mean, sample_var,running_mean,running_var = batchnorm_forward(hidden_before_relu, gamma, beta, 'train',running_mean,running_var)
    print("After batchnorm_forward, H = ")
    print(hidden_before_relu)
    print("After batchnorm_forward, H_normalized = ")
    print(x_normalized)
    print("After batchnorm_forward, mean = ")
    print(sample_mean)
    print("After batchnorm_forward, var = ")
    print(sample_var)
    print("After batchnorm_forward, mean_cache = ")
    print(running_mean)
    print("After batchnorm_forward, var_cache = ")
    print(running_var)
    print("After batchnorm_forward, gamma = ")
    print(gamma)
    print("After batchnorm_forward, beta = ")
    print(beta)
    print("After batchnorm_forward, running_mean = ")
    print(running_mean)
    print("After batchnorm_forward, running_var = ")
    print(running_var)
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
    print("Before batchnorm_backward, dH = ")
    print(dhidden)
    dhidden, dgamma, dbeta = batchnorm_backward(dhidden, x_normalized, gamma, beta, sample_mean, sample_var, hidden, 1e-5)
    print("After batchnorm_backward, dH = ")
    print(dhidden)
    print("After batchnorm_backward, dgamma = ")
    print(dgamma)
    print("After batchnorm_backward, dbeta = ")
    print(dbeta)
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

    gamma += -step_size*dgamma
    beta += -step_size*dbeta
    """
    vW = mu*vW + step_size*dW
    W -= vW
    vb = mu*vb + step_size*db
    b -= vb
    vW2 = mu*vW2 + step_size*dW2
    W2 -= vW2
    vb2 = mu*vb2 + step_size*db2
    b2 -= vb2
    """
    
    print("After update")
    print("W = ")
    print(W)
    print("b = ")
    print(b)
    print("W2 = ")
    print(W2)
    print("b2 = ")
    print(b2)


hidden_before_relu = np.dot(X,W) + b
hidden_before_relu, _ = batchnorm_forward(hidden_before_relu, gamma, beta, 'test',running_mean,running_var)
hidden = np.maximum(0,np.dot(X,W) + b)
test_scores = np.dot(hidden, W2) + b2
predicted = np.argmax(test_scores, axis=1)
print("Predicted socres:")
print(predicted)
print("Correctness: %f" % (np.mean(predicted == y)))


