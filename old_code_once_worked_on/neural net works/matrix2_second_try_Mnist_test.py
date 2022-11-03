import numpy
import cupy
import numpy as np
from numpy import array, ones, zeros, matmul, random
import os, time, math
from math import sqrt
import matrix2_second_try_tool as tool
from variables import v_w1,v_b1,v_w2,v_b2,v_w3,v_b3,v_w4,v_b4,v_w5,v_b5,save_temp_v,read_temp_v

Cupy = False

study_rate = 0.03
cons = 1
Exp = -3
Loop = 10000*6
Peek_Loop = Loop/200
Moment = 0.9
AdaDelta = 0.999
Value = True
Moment_boolean = Value
Batch = 20

tool.resize_pic()
ans_pic,y,_y = tool.read_pic()
print(ans_pic)
print(y)
print(_y)
if(Value):
    if(not Cupy):
        w1=v_w1();b1=v_b1();w2=v_w2();b2=v_b2();w3=v_w3();b3=v_b3();w4=v_w4();b4=v_b4();w5=v_w5();b5=v_b5();
    else:
        w1=np.array(v_w1());b1=np.array(v_b1());w2=np.array(v_w2());b2=np.array(v_b2());
        w3=np.array(v_w3());b3=np.array(v_b3());w4=np.array(v_w4());b4=np.array(v_b4());
        w5=np.array(v_w5());b5=np.array(v_b5());

count = 0
for i in range(ans_pic.shape[0]):
    ans_id = i
    x1 = tool.relu(tool.convolute(ans_pic[ans_id],w1)+b1)
    x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
    x2 = tool.relu(tool.convolute(x1,w2)+b2)
    x2_shape2 = x2.shape
    x2 = tool.conv3D_step(x2,(ones((x2.shape[0],2,2))/4),2,2)
    x2_shape = x2.shape
    x2 = x2.reshape((1,-1))

    x3 = tool.relu(matmul(x2,w3)+b3)
    x4 = tool.relu(matmul(x3,w4)+b4)
    x5 = tool.relu(matmul(x4,w5)+b5)

    C = (tool.softmax(x5)-y[ans_id])**2
    c = abs(tool.softmax(x5/10)-y[ans_id])
    print("\n",i,"\n","\nx5:",x5,"\nsoftmax(x5):",tool.softmax(x5/10),"\ny:",y[ans_id],"\nc:",c,"\nlabel:",_y[ans_id])
    if((c<0.5).all()):
        count = count + 1
    else:
        print("********************************\n")

print(count)















