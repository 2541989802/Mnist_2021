import numpy as np
from numpy import array, ones, zeros, matmul, random
import os, time
import matrix2_second_try_tool as tool


study_rate = 1000
cons = 1
Exp = -2
Loop = 40000
Peek_Loop = Loop/10
Momemont = 0.5

ans_pic = []
y = []
ans_pic.append(random.random((1,28,28)));
ans_pic.append(random.random((1,28,28)));

for i in range(len(ans_pic)):
    _y = zeros((1,len(ans_pic)))
    _y[0,i] = 1
    y.append(_y)

w1 = random.random((2,ans_pic[0].shape[0],3,3))*10**Exp
b1 = random.random((w1.shape[0],1,1))*0

w2 = random.random((2,w1.shape[0],3,3))*10**Exp
b2 = random.random((w2.shape[0],1,1))*0

x1 = tool.sigmoid(tool.convolute(ans_pic[0],w1)+b1)
x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
x2_shape = x2.shape
x2 = x2.reshape((1,-1))

w3 = random.random((x2.shape[1],50))*10**Exp
b3 = random.random((1,w3.shape[1]))
w4 = random.random((w3.shape[1],10))*10**Exp
b4 = random.random((1,w4.shape[1]))
w5 = random.random((w4.shape[1],y[0].shape[1]))*10**Exp
b5 = random.random((1,w5.shape[1]))

dCdw5=0;dCdb5=0;dCdw4=0;dCdb4=0;dCdw3=0;dCdb3=0;dCdw2=0;dCdb2=0;dCdw1=0;dCdb1=0;
for interior in range(Loop):
    ans_id = (int)(random.random()*len(ans_pic))

    x1 = tool.sigmoid(tool.convolute(ans_pic[ans_id],w1)+b1)
    x1_shape = x1.shape
    x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
    x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
    x2_shape = x2.shape
    x2 = x2.reshape((1,-1))

    x3 = tool.sigmoid(matmul(x2,w3)+b3)
    x4 = tool.sigmoid(matmul(x3,w4)+b4)
    x5 = tool.sigmoid(matmul(x4,w5)+b5)    

    C = (x5-y[ans_id])**2

    dCdx5 = (x5-y[ans_id])*cons
    dCdb5 = dCdx5*tool.d_sigmoid(x5) +dCdb5*Momemont
    dCdw5 = matmul(x4.T,dCdb5) +dCdw5*Momemont
    dCdx4 = matmul(dCdb5,w5.T)
    dCdb4 = dCdx4*tool.d_sigmoid(x4) +dCdb4*Momemont
    dCdw4 = matmul(x3.T,dCdb4) +dCdw4*Momemont
    dCdx3 = matmul(dCdb4,w4.T)
    dCdb3 = dCdx3*tool.d_sigmoid(x3) +dCdb3*Momemont
    dCdw3 = matmul(x2.T,dCdb3) +dCdw3*Momemont
    dCdx2 = matmul(dCdb3,w3.T).reshape(x2_shape)
    dCdx2 = dCdx2*tool.d_sigmoid(x2.reshape(x2_shape))
    dCdb2 = sum(dCdx2,0) +dCdb2*Momemont
    dCdw2 = tool.dw_convolute(dCdx2,x1,w2) +dCdw2*Momemont
    dCdx1 = tool.dx_convolute(dCdx2,x1,w2)
    dCdx1 = dCdx1*tool.d_sigmoid(x1)
    dCdx1 = tool.expand3D(dCdx1,ones((x1_shape[0],2,2)),x1_shape[1],x1_shape[2])
    dCdb1 = sum(dCdx1,0) +dCdb1*Momemont
    dCdw1 = tool.dw_convolute(dCdx1,ans_pic[ans_id],w1) +dCdw1*Momemont

    w1 = w1-study_rate*dCdw1
    b1 = b1-study_rate*dCdb1
    w2 = w2-study_rate*dCdw2
    b2 = b2-study_rate*dCdb2
    w3 = w3-study_rate*dCdw3
    b3 = b3-study_rate*dCdb3
    if(interior%Peek_Loop==0):
        if((dCdw1==0).all()):
            print(x1)
        print("\ninterior:\n",interior)
        print("\nC:\n",C)

count = 0
for i in range(20):
    ans_id = (int)(random.random()*len(ans_pic))

    x1 = tool.sigmoid(tool.convolute(ans_pic[ans_id],w1)+b1)
    x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
    x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
    x2_shape = x2.shape
    x2 = x2.reshape((1,-1))

    x3 = tool.sigmoid(matmul(x2,w3)+b3)
    x4 = tool.sigmoid(matmul(x3,w4)+b4)
    x5 = tool.sigmoid(matmul(x4,w5)+b5)

    C = (x5-y[ans_id])**2
    c = abs(x5-y[ans_id])
    print("\nx5:",x5,"\ny:",y[ans_id],"\nc:",c)
    if((c<0.5).all()):
        count = count + 1

print(count)



















