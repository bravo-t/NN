#!/bin/env python3

import numpy as np

def xrange(N):
    return range(N)

def conv_forward_naive(x, w, b, conv_param):
    stride, pad = conv_param['stride'], conv_param['pad']
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant') #补零
    H_new = int(1 + (H + 2 * pad - HH) / stride)
    W_new = int(1 + (W + 2 * pad - WW) / stride)
    s = stride
    print(H_new)
    print(W_new)
    out = np.zeros((N, F, H_new, W_new))


    for i in xrange(N):       # ith image    
        for f in xrange(F):   # fth filter        
            for j in xrange(H_new):            
                for k in xrange(W_new):                
                    out[i, f, j, k] = np.sum(x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] * w[f]) + b[f]#对应位相乘

    cache = (x, w, b, conv_param)

    return out, cache


def conv_backward_naive(dout, cache):
    x, w, b, conv_param = cache
    pad = conv_param['pad']
    stride = conv_param['stride']
    F, C, HH, WW = w.shape
    N, C, H, W = x.shape
    H_new = 1 + (H + 2 * pad - HH) / stride
    W_new = 1 + (W + 2 * pad - WW) / stride

    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)

    s = stride
    x_padded = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')
    dx_padded = np.pad(dx, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant')

    for i in xrange(N):       # ith image    
        for f in xrange(F):   # fth filter        
            for j in xrange(H_new):            
                for k in xrange(W_new):                
                    window = x_padded[i, :, j*s:HH+j*s, k*s:WW+k*s]
                    db[f] += dout[i, f, j, k]                
                    dw[f] += window * dout[i, f, j, k]                
                    dx_padded[i, :, j*s:HH+j*s, k*s:WW+k*s] += w[f] * dout[i, f, j, k]#上面的式子，关键就在于+号

    # Unpad
    dx = dx_padded[:, :, pad:pad+H, pad:pad+W]

    return dx, dw, db

def max_pool_forward_naive(x, pool_param):
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) / s
    W_new = 1 + (W - WW) / s
    out = np.zeros((N, C, H_new, W_new))
    for i in xrange(N):    
        for j in xrange(C):        
            for k in xrange(H_new):            
                for l in xrange(W_new):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s] 
                    out[i, j, k, l] = np.max(window)

    cache = (x, pool_param)

    return out, cache


def max_pool_backward_naive(dout, cache):
    x, pool_param = cache
    HH, WW = pool_param['pool_height'], pool_param['pool_width']
    s = pool_param['stride']
    N, C, H, W = x.shape
    H_new = 1 + (H - HH) / s
    W_new = 1 + (W - WW) / s
    dx = np.zeros_like(x)
    for i in xrange(N):    
        for j in xrange(C):        
            for k in xrange(H_new):            
                for l in xrange(W_new):                
                    window = x[i, j, k*s:HH+k*s, l*s:WW+l*s]                
                    m = np.max(window)               #获得之前的那个值，这样下面只要windows==m就能得到相应的位置
                    dx[i, j, k*s:HH+k*s, l*s:WW+l*s] = (window == m) * dout[i, j, k, l]

    return dx

def print_np_array(a):
    n,c,h,w = a.shape
    for i in range(n):
        print("%d %d %d" % (c,h,w))
        for j in range(c):
            for k in range(h):
                for l in range(w):
                    # stupid python
                    print("%f " % a[i,j,k,l], end="")
                print("")

conv_param = {"stride":2, "pad":0}
pool_param = {"pool_width":2, "pool_height":2, "stride":2}
x = np.random.rand(1,3,8,8)
print_np_array(x)
f = np.random.rand(1,3,2,2)
print_np_array(f)
b = np.random.rand(1,1,1,1)
print_np_array(b)
out, cache = conv_forward_naive(x,f,b,conv_param)
print_np_array(out)