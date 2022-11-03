import numpy as np
from numpy import array, ones, zeros, matmul, random
import os, time
import matrix2_second_try_tool as tool


study_rate = 1
Loop = 40000
Peek_Loop = Loop/10

ans_pic = []
y = []
ans_pic.append(random.random((2,7,7)));y.append(array([[0]]))
ans_pic.append(random.random((2,7,7)));y.append(array([[1]]))

w1 = random.random((2,2,3,3))
b1 = random.random((w1.shape[0],1,1))
w2 = random.random((2,w1.shape[0],3,3))
b2 = random.random((w2.shape[0],1,1))

x1 = tool.sigmoid(tool.convolute(ans_pic[0],w1)+b1)
x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
x2_shape = x2.shape
x2 = x2.reshape((1,-1))

w3 = random.random((x2.shape[1],1))
b3 = random.random((1,w3.shape[1]))

for interior in range(Loop):
    ans_id = (int)(random.random()*len(ans_pic))

    x1 = tool.sigmoid(tool.convolute(ans_pic[ans_id],w1)+b1)
    x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
    x2_shape = x2.shape
    x2 = x2.reshape((1,-1))

    x3 = tool.sigmoid(matmul(x2,w3)+b3)

    C = (x3-y[ans_id])**2

    dCdx3 = (x3-y[ans_id])
    dCdb3 = dCdx3*tool.d_sigmoid(x3)
    dCdw3 = matmul(x2.T,dCdb3)
    dCdx2 = matmul(dCdb3,w3.T).reshape(x2_shape)
    dCdx2 = dCdx2*tool.d_sigmoid(x2.reshape(x2_shape))
    dCdb2 = sum(dCdx2,0)
    dCdw2 = tool.dw_convolute(dCdx2,x1,w2)
    dCdx1 = tool.dx_convolute(dCdx2,x1,w2)
    dCdx1 = dCdx1*tool.d_sigmoid(x1)
    dCdb1 = sum(dCdx1,0)
    dCdw1 = tool.dw_convolute(dCdx1,ans_pic[ans_id],w1)

    w1 = w1-study_rate*dCdw1
    b1 = b1-study_rate*dCdb1
    w2 = w2-study_rate*dCdw2
    b2 = b2-study_rate*dCdb2
    w3 = w3-study_rate*dCdw3
    b3 = b3-study_rate*dCdb3
    if(interior%Peek_Loop==0):
        print("\ninterior:\n",interior)
        print("\nC:\n",C)

count = 0
for i in range(20):
    ans_id = (int)(random.random()*len(ans_pic))

    x1 = tool.sigmoid(tool.convolute(ans_pic[ans_id],w1)+b1)
    x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
    x2_shape = x2.shape
    x2 = x2.reshape((1,-1))

    x3 = tool.sigmoid(matmul(x2,w3)+b3)

    C = (x3-y[ans_id])**2
    c = abs(x3-y[ans_id])
    print("x3:",x3,"\ty:",y[ans_id],"\tc:",c)
    if((c<0.5).all()):
        count = count + 1

print(count)



















