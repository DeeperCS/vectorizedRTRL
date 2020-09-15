'''
RTRL for Bus Driver Problem

### Detect the first 'b' after 'a' in a sequence (4 possible characters including 'a b c d')

Scalarized form (hidden_dim: 4, time cost: around 2.0 s)

Time cost: 1.9623632431030273
Test:1000/1000
'''

import numpy as np
import time
import matplotlib.pyplot as plt

f = lambda x: 1 / (1+np.exp(-x))
delta = lambda i, j: i==j

dim_hidden = 4
dim_input = 4

W = -1 + 2 * np.random.rand(dim_hidden, (dim_hidden + dim_input + 1))
p1 = np.zeros_like(W)
p2 = np.zeros_like(W)

preInd = -1
alpha = 5
maxIter = 3000
y = np.zeros((dim_hidden, 1))
err_arr = []
time_begin = time.time()
for t in range(maxIter):
    x = np.zeros((dim_input,1))
    currInd = np.random.randint(4)  # 0:a, 1:b, 2:c, 3:d
    if t == 0:
        currInd = 0
    x[currInd] = 1
    if preInd == 0 and currInd == 1:  # previously seen a, currently see b
        d = 1  # target output 1
        preInd = -1  # reset preInd
    else:
        d = 0  # target output 0
        preInd = -1  # reset preInd
        if currInd == 0:  # record a:0
            preInd = currInd

    bias = np.ones((1, 1))
    z = np.concatenate([y, x, bias], axis=0)
    
    s = W.dot(z)
    y = f(s)
    
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            q1 = f(s[0, 0]) * (1-f(s[0, 0])) * (W[0,0] * p1[i,j] + W[0,1] * p2[i,j] + delta(i, 0) * z[j])
            q2 = f(s[1, 0]) * (1-f(s[1, 0])) * (W[1,0] * p1[i,j] + W[1,1] * p2[i,j] + delta(i, 1) * z[j])
            p1[i, j] = q1
            p2[i, j] = q2
    
    err = d - y[-1]
    W = W + alpha * err * p2

    # print(err)
    err_arr.append(abs(err))
    # assert(0)
plt.plot(err_arr)

print("Time cost:", time.time()-time_begin)
# Test
x = np.zeros((dim_input,1))
currInd = 0
x[currInd] = 1
preInd = currInd

y = np.zeros((dim_hidden,1))
bias = np.ones((1, 1))
z = np.concatenate([y, x, bias], axis=0)
s = W.dot(z)
y = f(s)
passed = 0
abN = 0

test_len = 1000
for t in range(test_len):
    x = np.zeros((dim_input,1))
    currInd = np.random.randint(4)  # 0:a, 1:b, 2:c, 3:d
    x[currInd] = 1
    if currInd == 1 and preInd == 0:  # previously seen a, currently see b
        abN += 1  # target output 1
        d = 1
        preInd = -1  # reset preInd
    else:
        d = 0  # target output 0
        preInd = -1  # reset preInd
        if currInd == 0:  # record a:0
            preInd = currInd
    
    bias = np.ones((1, 1))
    z = np.concatenate([y, x, bias], axis=0)
    
    s = W.dot(z)
    y = f(s)
    if (y[1][0]>=0.5) == d:
        passed += 1
        
print("Test:{}/{}".format(passed, test_len))