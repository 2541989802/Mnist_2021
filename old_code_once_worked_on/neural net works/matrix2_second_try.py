import numpy
import cupy
import numpy as np
from numpy import array, ones, zeros, matmul, random
import os, time, math
from math import sqrt
import matrix2_second_try_tool as tool
from variables import v_w1,v_b1,v_w2,v_b2,v_w3,v_b3,v_w4,v_b4,v_w5,v_b5,save_temp_v,read_temp_v

Cupy = False

study_rate = 0.1
cons = 1
Exp = -3
Loop = 10000
Peek_Loop = Loop/200
Moment = 0.9
AdaDelta = 0.999
Value = False
Ans = False
Moment_boolean = False
Batch = 20

ans_pic = []
y = []

size = 28
ans_pic.append(random.random((1,size,size)));
ans_pic.append(random.random((1,size,size)));
ans_pic.append(random.random((1,size,size)));
ans_pic.append(random.random((1,size,size)));
ans_pic.append(random.random((1,size,size)));
if(Ans):
    for i in range(len(ans_pic)):
        name = "ans_"+str(i)+".txt"
        ans_pic[i] = read_temp_v(name)if(not Cupy) else np.array(read_temp_v(name))
        print(ans_pic[i])

for i in range(len(ans_pic)):
    _y = zeros((1,len(ans_pic)))
    _y[0,i] = 1

    y.append(_y)

w1 = random.random((6,ans_pic[0].shape[0],5,5))*10**Exp
w1 = w1*random.randint(-1,2,(w1.shape))
b1 = random.random((w1.shape[0],1,1))*0
w2 = random.random((16,w1.shape[0],5,5))*10**Exp
w2 = w2*random.randint(-1,2,(w2.shape))
b2 = random.random((w2.shape[0],1,1))*0

x1 = tool.sigmoid(tool.convolute(ans_pic[0],w1)+b1)
x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
x2 = tool.conv3D_step(x2,(ones((x2.shape[0],2,2))/4),2,2)
x2_shape = x2.shape
x2 = x2.reshape((1,-1))

w3 = random.random((x2.shape[1],120))*10**Exp
b3 = random.random((1,w3.shape[1]))
w4 = random.random((w3.shape[1],84))*10**Exp
b4 = random.random((1,w4.shape[1]))
w5 = random.random((w4.shape[1],y[0].shape[1]))*10**Exp
b5 = random.random((1,w5.shape[1]))

dw5=array([w5*0,w5*0]);db5=array([b5*0,b5*0]);
dw4=array([w4*0,w4*0]);db4=array([b4*0,b4*0]);
dw3=array([w3*0,w3*0]);db3=array([b3*0,b3*0]);
dw2=array([w2*0,w2*0]);db2=array([b2*0,b2*0]);
dw1=array([w1*0,w1*0]);db1=array([b1*0,b1*0]);
if(Moment_boolean):
    dw5=read_temp_v("v_dw5.txt");dw4=read_temp_v("v_dw4.txt");dw3=read_temp_v("v_dw3.txt");dw2=read_temp_v("v_dw2.txt");dw1=read_temp_v("v_dw1.txt");
    db5=read_temp_v("v_db5.txt");db4=read_temp_v("v_db4.txt");db3=read_temp_v("v_db3.txt");db2=read_temp_v("v_db2.txt");db1=read_temp_v("v_db1.txt");

if(Value):
    if(not Cupy):
        w1=v_w1();b1=v_b1();w2=v_w2();b2=v_b2();w3=v_w3();b3=v_b3();w4=v_w4();b4=v_b4();w5=v_w5();b5=v_b5();
    else:
        w1=np.array(v_w1());b1=np.array(v_b1());w2=np.array(v_w2());b2=np.array(v_b2());
        w3=np.array(v_w3());b3=np.array(v_b3());w4=np.array(v_w4());b4=np.array(v_b4());
        w5=np.array(v_w5());b5=np.array(v_b5());

time_start = time.process_time()
for interior in range(Loop):
    _dCdb5=0;_dCdw5=0;_dCdb4=0;_dCdw4=0;_dCdb3=0;_dCdw3=0;_dCdb2=0;_dCdw2=0;_dCdb1=0;_dCdw1=0;
    tw1=w1;tw2=w2;tw3=w3;tw4=w4;tw5=w5;
    show_id = (int)(random.random()*len(ans_pic));_x5=0;_C=0;
    for batch_i in range(Batch):
        #ans_id = (int)(random.random()*len(ans_pic))
        ans_id = batch_i%len(ans_pic)

        x1 = tool.sigmoid(tool.convolute(ans_pic[ans_id],w1)+b1)
        x1_shape = x1.shape
        x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
        
        x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
        x2_shape2 = x2.shape
        x2 = tool.conv3D_step(x2,(ones((x2.shape[0],2,2))/4),2,2)
        x2_shape = x2.shape
        x2 = x2.reshape((1,-1))

        x3 = tool.sigmoid(matmul(x2,w3)+b3)
        x4 = tool.sigmoid(matmul(x3,w4)+b4)
        x5 = tool.sigmoid(matmul(x4,w5)+b5)    

        C = (x5-y[ans_id])**2
        if(show_id==batch_i):
           _C = C
           _x5 = x5

        #dCdx5 = (x5-y[ans_id])*cons
        dCdx5 = (tool.softmax(x5)-y[ans_id])*cons
        dCdb5 = dCdx5*tool.d_sigmoid(x5);    _dCdb5=_dCdb5 + dCdb5/Batch;
        dCdw5 = matmul(x4.T,dCdb5); _dCdw5=_dCdw5 + dCdw5/Batch;

        dCdx4 = matmul(dCdb5,w5.T)
        dCdb4 = dCdx4*tool.d_sigmoid(x4);   _dCdb4=_dCdb4 + dCdb4/Batch;
        dCdw4 = matmul(x3.T,dCdb4); _dCdw4=_dCdw4 + dCdw4/Batch;

        dCdx3 = matmul(dCdb4,w4.T)
        dCdb3 = dCdx3*tool.d_sigmoid(x3);   _dCdb3=_dCdb3 + dCdb3/Batch;
        dCdw3 = matmul(x2.T,dCdb3); _dCdw3=_dCdw3 + dCdw3/Batch;

        dCdx2 = matmul(dCdb3,w3.T).reshape(x2_shape)
        dCdx2 = dCdx2*tool.d_sigmoid(x2.reshape(x2_shape))#/4
        dCdx2 = tool.expand3D(dCdx2,ones((x2_shape2[0],2,2)),x2_shape2[1],x2_shape2[2])
        dCdb2 = np.sum(dCdx2,axis=(1,2),keepdims=True);   _dCdb2=_dCdb2 + dCdb2/Batch;
        dCdw2 = tool.dw_convolute(dCdx2,x1,w2); _dCdw2=_dCdw2 + dCdw2/Batch;

        dCdx1 = tool.dx_convolute(dCdx2,x1,w2)
        dCdx1 = dCdx1*tool.d_sigmoid(x1)
        dCdx1 = tool.expand3D(dCdx1,ones((x1_shape[0],2,2)),x1_shape[1],x1_shape[2])
        dCdb1 = np.sum(dCdx1,axis=(1,2),keepdims=True);   _dCdb1=_dCdb1 + dCdb1/Batch;
        dCdw1 = tool.dw_convolute(dCdx1,ans_pic[ans_id],w1);    _dCdw1=_dCdw1 + dCdw1/Batch;


    """print("\nw5.shape:",w5.shape);print("\nw4.shape:",w4.shape);print("\nw3.shape:",w3.shape);print("\nw2.shape:",w2.shape);print("\nw1.shape:",w1.shape);
    print("\n_dCdw5.shape:",_dCdw5.shape);print("\n_dCdw4.shape:",_dCdw4.shape);print("\n_dCdw3.shape:",_dCdw3.shape);print("\n_dCdw2.shape:",_dCdw2.shape);print("\n_dCdw1.shape:",_dCdw1.shape);
    print("\ndw5.shape:",dw5[0].shape);print("\ndw4.shape:",dw4[0].shape);print("\ndw3.shape:",dw3[0].shape);print("\ndw2.shape:",dw2[0].shape);print("\ndw1.shape:",dw1[0].shape);
    print("\nb5.shape:",b5.shape);print("\nb4.shape:",b4.shape);print("\nb3.shape:",b3.shape);print("\nb2.shape:",b2.shape);print("\nb1.shape:",b1.shape);
    print("\n_dCdb5.shape:",_dCdb5.shape);print("\n_dCdb4.shape:",_dCdb4.shape);print("\n_dCdb3.shape:",_dCdb3.shape);print("\n_dCdb2.shape:",_dCdb2.shape);print("\n_dCdb1.shape:",_dCdb1.shape);
    print("\ndb5.shape:",db5[0].shape);print("\ndb4.shape:",db4[0].shape);print("\ndb3.shape:",db3[0].shape);print("\ndb2.shape:",db2[0].shape);print("\ndb1.shape:",db1[0].shape);
    """

    dw5[0] = Moment*dw5[0] + (1-Moment)*_dCdw5
    dw5[1] = AdaDelta*dw5[1] + (1-AdaDelta)*(_dCdw5**2)
    dw4[0] = Moment*dw4[0] + (1-Moment)*_dCdw4
    dw4[1] = AdaDelta*dw4[1] + (1-AdaDelta)*(_dCdw4**2)
    dw3[0] = Moment*dw3[0] + (1-Moment)*_dCdw3
    dw3[1] = AdaDelta*dw3[1] + (1-AdaDelta)*(_dCdw3**2)
    dw2[0] = Moment*dw2[0] + (1-Moment)*_dCdw2
    dw2[1] = AdaDelta*dw2[1] + (1-AdaDelta)*(_dCdw2**2)
    dw1[0] = Moment*dw1[0] + (1-Moment)*_dCdw1
    dw1[1] = AdaDelta*dw1[1] + (1-AdaDelta)*(_dCdw1**2)

    db5[0] = Moment*db5[0] + (1-Moment)*_dCdb5
    db5[1] = AdaDelta*db5[1] + (1-AdaDelta)*(_dCdb5**2)
    db4[0] = Moment*db4[0] + (1-Moment)*_dCdb4
    db4[1] = AdaDelta*db4[1] + (1-AdaDelta)*(_dCdb4**2)
    db3[0] = Moment*db3[0] + (1-Moment)*_dCdb3
    db3[1] = AdaDelta*db3[1] + (1-AdaDelta)*(_dCdb3**2)
    db2[0] = Moment*db2[0] + (1-Moment)*_dCdb2
    db2[1] = AdaDelta*db2[1] + (1-AdaDelta)*(_dCdb2**2)
    db1[0] = Moment*db1[0] + (1-Moment)*_dCdb1
    db1[1] = AdaDelta*db1[1] + (1-AdaDelta)*(_dCdb1**2)

    w1 = w1-study_rate*dw1[0]*(1-Moment)/tool.m_h(10e-16+dw1[1]/(1-AdaDelta),sqrt)
    b1 = b1-study_rate*db1[0]*(1-Moment)/tool.m_h(10e-16+db1[1]/(1-AdaDelta),sqrt)
    w2 = w2-study_rate*dw2[0]*(1-Moment)/tool.m_h(10e-16+dw2[1]/(1-AdaDelta),sqrt)
    b2 = b2-study_rate*db2[0]*(1-Moment)/tool.m_h(10e-16+db2[1]/(1-AdaDelta),sqrt)
    w3 = w3-study_rate*dw3[0]*(1-Moment)/tool.m_h(10e-16+dw3[1]/(1-AdaDelta),sqrt)
    b3 = b3-study_rate*db3[0]*(1-Moment)/tool.m_h(10e-16+db3[1]/(1-AdaDelta),sqrt)
    w4 = w4-study_rate*dw4[0]*(1-Moment)/tool.m_h(10e-16+dw4[1]/(1-AdaDelta),sqrt)
    b4 = b4-study_rate*db4[0]*(1-Moment)/tool.m_h(10e-16+db4[1]/(1-AdaDelta),sqrt)
    w5 = w5-study_rate*dw5[0]*(1-Moment)/tool.m_h(10e-16+dw5[1]/(1-AdaDelta),sqrt)
    b5 = b5-study_rate*db5[0]*(1-Moment)/tool.m_h(10e-16+db5[1]/(1-AdaDelta),sqrt)
    if(interior%Peek_Loop==0):
        time_d = time.process_time()-time_start
        minute = time_d/60
        hour = minute/60
        print("\ntime: ",(int)(hour%60),':',(int)(minute%60),';',time_d%60,'\n')
        if(((w1-tw1)==0).all()):
            print("********** warning, w1 doesn't change **********")
        print("\ninterior:\n",interior)
        print("\nC:\n",_C)
        print("\nx5:\n",_x5)
        print("\nsoftmax(x5):\n",tool.softmax(_x5))
        print("\nans_pic:\n",y[show_id])
        print("\nnp.sum(abs(w1-tw1)):",np.sum(abs(w1-tw1)))
        print("\nnp.sum(abs(w2-tw2)):",np.sum(abs(w2-tw2)))
        print("\nnp.sum(abs(w3-tw3)):",np.sum(abs(w3-tw3)))
        print("\nnp.sum(abs(w4-tw4)):",np.sum(abs(w4-tw4)))
        print("\nnp.sum(abs(w5-tw5)):",np.sum(abs(w5-tw5)))

count = 0
for i in range(20):
    ans_id = (int)(random.random()*len(ans_pic))

    x1 = tool.sigmoid(tool.convolute(ans_pic[ans_id],w1)+b1)
    x1 = tool.conv3D_step(x1,(ones((x1.shape[0],2,2))/4),2,2)
    x2 = tool.sigmoid(tool.convolute(x1,w2)+b2)
    x2_shape2 = x2.shape
    x2 = tool.conv3D_step(x2,(ones((x2.shape[0],2,2))/4),2,2)
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

save_temp_v(dw5,"v_dw5.txt")
save_temp_v(db5,"v_db5.txt")
save_temp_v(dw4,"v_dw4.txt")
save_temp_v(db4,"v_db4.txt")
save_temp_v(dw3,"v_dw3.txt")
save_temp_v(db3,"v_db3.txt")
save_temp_v(dw2,"v_dw2.txt")
save_temp_v(db2,"v_db2.txt")
save_temp_v(dw1,"v_dw1.txt")
save_temp_v(db1,"v_db1.txt")
for i in range(len(ans_pic)):
    name = "ans_"+str(i)+".txt"
    if(not Cupy):
        save_temp_v(ans_pic[i],name)
    else:
        save_temp_v(np.asnumpy(ans_pic[i]),name)















