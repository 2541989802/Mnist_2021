import numpy
import cupy
import numpy as np
from numpy import array, ones, zeros, matmul, random
import os, time, math
from math import sqrt
import matrix2_second_try_tool as tool
from variables import v_w1,v_b1,v_w2,v_b2,v_w3,v_b3,v_w4,v_b4,v_w5,v_b5,save_temp_v,read_temp_v

Cupy = False

study_rate = 1
cons = 1
Exp = -3
Loop = 20000
Peek_Loop = Loop/100
Moment = 0.9
AdaDelta = 0.999
Value = False

ans_pic = []
y = []

ans_pic.append(random.random((1,28,28)));
ans_pic.append(random.random((1,28,28)));
ans_pic.append(random.random((1,28,28)));
ans_pic.append(random.random((1,28,28)));
ans_pic.append(random.random((1,28,28)));
if(Value):
    for i in range(len(ans_pic)):
        name = "ans_"+str(i)+".txt"
        ans_pic[i] = read_temp_v(name)if(not Cupy) else np.array(read_temp_v(name))

for i in range(len(ans_pic)):
    _y = zeros((1,len(ans_pic)))
    _y[0,i] = 1
    y.append(_y)

w1 = random.random((5,ans_pic[0].shape[0],3,3))*10**Exp
b1 = random.random((w1.shape[0],1,1))*0

w2 = random.random((5,w1.shape[0],3,3))*10**Exp
b2 = random.random((w2.shape[0],1,1))*0

x1 = tool.sigmoid(tool.convolute(ans_pic[0],w1)+b1)
x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
x2_shape = x2.shape
x2 = x2.reshape((1,-1))

w3 = random.random((x2.shape[1],10))*10**Exp
b3 = random.random((1,w3.shape[1]))
w4 = random.random((w3.shape[1],10))*10**Exp
b4 = random.random((1,w4.shape[1]))
w5 = random.random((w4.shape[1],y[0].shape[1]))*10**Exp
b5 = random.random((1,w5.shape[1]))

dw5=[w5*0,w5*0];db5=[b5*0,b5*0];
dw4=[w4*0,w4*0];db4=[b4*0,b4*0];
dw3=[w3*0,w3*0];db3=[b3*0,b3*0];
dw2=[w2*0,w2*0];db2=[b2*0,b2*0];
dw1=[w1*0,w1*0];db1=[b1*0,b1*0];

if(Value):
    if(not Cupy):
        w1=v_w1();b1=v_b1();w2=v_w2();b2=v_b2();w3=v_w3();b3=v_b3();w4=v_w4();b4=v_b4();w5=v_w5();b5=v_b5();
    else:
        w1=np.array(v_w1());b1=np.array(v_b1());w2=np.array(v_w2());b2=np.array(v_b2());
        w3=np.array(v_w3());b3=np.array(v_b3());w4=np.array(v_w4());b4=np.array(v_b4());
        w5=np.array(v_w5());b5=np.array(v_b5());


time_start = time.process_time()
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

    #dCdx5 = (x5-y[ans_id])*cons
    dCdx5 = (tool.softmax(x5)-y[ans_id])*cons
    dCdb5 = dCdx5[0]*tool.d_sigmoid(x5)
    dCdw5 = matmul(x4.T,dCdb5)

    dCdx4 = matmul(dCdb5,w5.T)
    dCdb4 = dCdx4*tool.d_sigmoid(x4)
    dCdw4 = matmul(x3.T,dCdb4)

    dCdx3 = matmul(dCdb4,w4.T)
    dCdb3 = dCdx3*tool.d_sigmoid(x3)*(1-Moment)
    dCdw3 = matmul(x2.T,dCdb3)

    dCdx2 = matmul(dCdb3,w3.T).reshape(x2_shape)
    dCdx2 = dCdx2*tool.d_sigmoid(x2.reshape(x2_shape))
    dCdb2 = sum(dCdx2,0)
    dCdw2 = tool.dw_convolute(dCdx2,x1,w2)

    dCdx1 = tool.dx_convolute(dCdx2,x1,w2)
    dCdx1 = dCdx1*tool.d_sigmoid(x1)
    dCdx1 = tool.expand3D(dCdx1,ones((x1_shape[0],2,2)),x1_shape[1],x1_shape[2])
    dCdb1 = sum(dCdx1,0)
    dCdw1 = tool.dw_convolute(dCdx1,ans_pic[ans_id],w1)

    dw5[0] = Moment*dw5[0] + (1-Moment)*dCdw5
    dw5[1] = AdaDelta*dw5[1] + (1-AdaDelta)*(dCdw5**2)
    dw4[0] = Moment*dw4[0] + (1-Moment)*dCdw4
    dw4[1] = AdaDelta*dw4[1] + (1-AdaDelta)*(dCdw4**2)
    dw3[0] = Moment*dw3[0] + (1-Moment)*dCdw3
    dw3[1] = AdaDelta*dw3[1] + (1-AdaDelta)*(dCdw3**2)
    dw2[0] = Moment*dw2[0] + (1-Moment)*dCdw2
    dw2[1] = AdaDelta*dw2[1] + (1-AdaDelta)*(dCdw2**2)
    dw1[0] = Moment*dw1[0] + (1-Moment)*dCdw1
    dw1[1] = AdaDelta*dw1[1] + (1-AdaDelta)*(dCdw1**2)

    db5[0] = Moment*db5[0] + (1-Moment)*dCdb5
    db5[1] = AdaDelta*db5[1] + (1-AdaDelta)*(dCdb5**2)
    db4[0] = Moment*db4[0] + (1-Moment)*dCdb4
    db4[1] = AdaDelta*db4[1] + (1-AdaDelta)*(dCdb4**2)
    db3[0] = Moment*db3[0] + (1-Moment)*dCdb3
    db3[1] = AdaDelta*db3[1] + (1-AdaDelta)*(dCdb3**2)
    db2[0] = Moment*db2[0] + (1-Moment)*dCdb2
    db2[1] = AdaDelta*db2[1] + (1-AdaDelta)*(dCdb2**2)
    db1[0] = Moment*db1[0] + (1-Moment)*dCdb1
    db1[1] = AdaDelta*db1[1] + (1-AdaDelta)*(dCdb1**2)

    w1 = w1-study_rate*dw1[0]*(1-Moment)/tool.m_h(10e-8+dw1[1]/(1-AdaDelta),sqrt)
    b1 = b1-study_rate*db1[0]*(1-Moment)/tool.m_h(10e-8+db1[1]/(1-AdaDelta),sqrt)
    w2 = w2-study_rate*dw2[0]*(1-Moment)/tool.m_h(10e-8+dw2[1]/(1-AdaDelta),sqrt)
    b2 = b2-study_rate*db2[0]*(1-Moment)/tool.m_h(10e-8+db2[1]/(1-AdaDelta),sqrt)
    w3 = w3-study_rate*dw3[0]*(1-Moment)/tool.m_h(10e-8+dw3[1]/(1-AdaDelta),sqrt)
    b3 = b3-study_rate*db3[0]*(1-Moment)/tool.m_h(10e-8+db3[1]/(1-AdaDelta),sqrt)
    w4 = w4-study_rate*dw4[0]*(1-Moment)/tool.m_h(10e-8+dw4[1]/(1-AdaDelta),sqrt)
    b4 = b4-study_rate*db4[0]*(1-Moment)/tool.m_h(10e-8+db4[1]/(1-AdaDelta),sqrt)
    w5 = w5-study_rate*dw5[0]*(1-Moment)/tool.m_h(10e-8+dw5[1]/(1-AdaDelta),sqrt)
    b5 = b5-study_rate*db5[0]*(1-Moment)/tool.m_h(10e-8+db5[1]/(1-AdaDelta),sqrt)
    if(interior%Peek_Loop==0):
        time_d = time.process_time()-time_start
        minute = time_d/60
        hour = minute/60
        print("time: ",(int)(hour%60),':',(int)(minute%60),';',time_d%60,'\n')
        if((dCdw1==0).all()):
            print(x1)
        print("\ninterior:\n",interior)
        print("\nC:\n",C)
        print("\nans_pic:\n",y[ans_id])

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

save_temp_v(w1,"v_w1.txt")
save_temp_v(w2,"v_w2.txt")
save_temp_v(w3,"v_w3.txt")
save_temp_v(w4,"v_w4.txt")
save_temp_v(w5,"v_w5.txt")
save_temp_v(b1,"v_b1.txt")
save_temp_v(b2,"v_b2.txt")
save_temp_v(b3,"v_b3.txt")
save_temp_v(b4,"v_b4.txt")
save_temp_v(b5,"v_b5.txt")
for i in range(len(ans_pic)):
    name = "ans_"+str(i)+".txt"
    if(not Cupy):
        save_temp_v(ans_pic[i],name)
    else:
        save_temp_v(np.asnumpy(ans_pic[i]),name)















