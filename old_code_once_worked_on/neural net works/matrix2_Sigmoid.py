import numpy as np
from numpy import array, exp, dot, matmul, matlib
import random, os
import numpy as ny

def conv2D(m, f):
    r = f.shape[0]
    c = f.shape[1]
    r2 = -1
    c2 = -1
    s = matlib.zeros((m.shape[0]-r+1,m.shape[1]-c+1))
    if(r>m.shape[0] or c>m.shape[1]):
        print("error:conv2D #1")
        os.system("pause")
    for i in range(r):
        for j in range(c):
            if(i-r+1!=0):
                r2 = i-r+1
            else:
                r2 = m.shape[0]
            if(j-c+1!=0):
                c2 = j-c+1
            else:
                c2 = m.shape[1]
            s = s + f[i,j]*m[i:r2,j:c2]
            #print(m[i:r2,j:c2])
    return s            

def conv3D(m, f):
    if(m.shape[0]!=f.shape[0]):
        print("error:conv3D #1")
        os.system("pause")
    s = np.zeros((m.shape[1]-f.shape[1]+1,m.shape[2]-f.shape[2]+1))
    for i in range(m.shape[0]):
        s = s + conv2D(m[i],f[i])
    return s

def m_half_r(r):
    _I = np.zeros(((int)((r+1)/2),r))
    for i in range((int)(r/2)):
        _I[i,2*i] = 1
        _I[i,2*i+1] = 1
    if((int)((r+1)/2)>r/2):
        _I[(int)(r/2),r-1] = 1
    return _I

def topTo1(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if(m[i,j]>1):
                m[i,j]=1
    return m

def pool2D(m):
    r = m.shape[0]
    c = m.shape[1]
    s = np.zeros((r,c))
    _I = np.zeros(((int)((r+1)/2),r))
    for i in range((int)(r/2)):
        _I[i,2*i] = 1
        _I[i,2*i+1] = 1
    if((int)((r+1)/2)>r/2):
        _I[(int)(r/2),r-1] = 2
    #print(_I)
    s = matmul(_I,m)
    #print(s)
    _I = np.zeros(((c,(int)((c+1)/2))))
    for i in range((int)(c/2)):
        _I[2*i,i] = 1
        _I[2*i+1,i] = 1
    if((int)((c+1)/2)>c/2):
        _I[c-1,(int)(c/2)] = 2
    #print(_I)
    s = matmul(s,_I)
    #print(s)
    s = s/4
    #s = topTo1(s)
    return s

def extend3DonXY(m,o):#m is half shrink of o on x and y, but same on z
    if(m.shape[0]!=o.shape[0] or m.shape[1]!=(int)((o.shape[1]+1)/2) or m.shape[2]!=(int)((o.shape[2]+1)/2)):
        print("error:extend3DonXY #1",m.shape,o.shape)
        os.system("pause")
    s = np.zeros(o.shape)
    for i in range(o.shape[0]):
        s[i] = matmul(m_half_r(s.shape[1]).T,matmul(m[i],m_half_r(s.shape[2])))
    return s

def sum3DtoZ(m):
    r = m.shape[1]
    c = m.shape[2]
    s = np.zeros((m.shape[0],1,1))
    for i in range(m.shape[0]):
        s[i] = matmul(np.ones((1,c)),matmul(m[i],np.ones((r,1))))
    return s

def convSum3DtoZ(m,f): #m.shape[0]==f.shape[0] and m.shape[1] > f.shape[1] and m.shape[2] > f.shape[2]
    r = f.shape[1]
    c = f.shape[2]
    #print("r:",r)
    #print("c:",c)
    r2 = -1
    c2 = -1
    s = np.zeros(f.shape)
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            if(i-r+1!=0):
                r2 = i-r+1
            else:
                r2 = m.shape[1]
            if(j-c+1!=0):
                c2 = j-c+1
            else:
                c2 = m.shape[2]
            t = sum3DtoZ(m[:,i:r2,j:c2])
            for k in range(f.shape[0]):
                s[k,i,j] = t[k,0,0]
    return s

def convTimesSum3DtoZ(m,f,o): #m times o then convolute, m.shape[0]==f.shape[0]==o.shape[0] and m.shape[1] > f.shape[1] and m.shape[2] > f.shape[2] and ,x1.shape[2]-w2.shape[3]+1)
    r = f.shape[1]
    c = f.shape[2]
    #print("r:",r)
    #print("c:",c)
    r2 = -1
    c2 = -1
    s = np.zeros(f.shape)
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            if(i-r+1!=0):
                r2 = i-r+1
            else:
                r2 = m.shape[1]
            if(j-c+1!=0):
                c2 = j-c+1
            else:
                c2 = m.shape[2]
            t = np.zeros((m[:,i:r2,j:c2]).shape)
            for k in range(t.shape[0]):
                t[k] = o
            t = sum3DtoZ(m[:,i:r2,j:c2]*t)
            for k in range(f.shape[0]):
                s[k,i,j] = t[k,0,0]
    return s

def multiConvSum(m,ks,o):#m times o then convolute, each 3D kernal compose a whole 4D filter, m.shape[0]=(ks[i].shape)[0],ks.shape[0]==o.shape[0], m.shape[1]-ks.shape[2]+1==o.shape[1],m.shape[2]-ks.shape[3]+1==o.shape[2]
    if(m.shape[1]-ks.shape[2]+1!=o.shape[1] or m.shape[2]-ks.shape[3]+1!=o.shape[2] or ks.shape[0]!=o.shape[0]):
        print("error:multiConvSum #1",m.shape,ks.shape,o.shape)
        os.system("pause")
    n = ks.shape[0]
    s = np.zeros(ks.shape)
    for i in range(n):
        s[i] = convTimesSum3DtoZ(m,ks[i],o[i])
    return s

def convTimesSum3DtoZ2(m,f,o): #convolutly f times o then add on zeros(m.shape), m.shape[0]==f.shape[0]==o.shape[0] and m.shape[1] > f.shape[1] and m.shape[2] > f.shape[2] and ,x1.shape[2]-w2.shape[3]+1)
    r = f.shape[1]
    c = f.shape[2]
    #print("r:",r)
    #print("c:",c)
    r2 = -1
    c2 = -1
    s = np.zeros(m.shape)
    for i in range(f.shape[1]):
        for j in range(f.shape[2]):
            if(i-r+1!=0):
                r2 = i-r+1
            else:
                r2 = m.shape[1]
            if(j-c+1!=0):
                c2 = j-c+1
            else:
                c2 = m.shape[2]
            for k in range(s.shape[0]):
                s[k,i:r2,j:c2]=s[k,i:r2,j:c2]+f[k,i,j]*o
    return s

def multiConvSum2(m,ks,o):#convolutly each ks times o then add on zeros(m.shape), each 3D kernal compose a whole 4D filter, m.shape[0]=(ks[i].shape)[0],ks.shape[0]==o.shape[0], m.shape[1]-ks.shape[2]+1==o.shape[1],m.shape[2]-ks.shape[3]+1==o.shape[2]
    if(m.shape[1]-ks.shape[2]+1!=o.shape[1] or m.shape[2]-ks.shape[3]+1!=o.shape[2] or ks.shape[0]!=o.shape[0]):
        print("error:multiConvSum2 #1",m.shape,ks.shape,o.shape)
        os.system("pause")
    n = ks.shape[0]
    s = np.zeros(m.shape)
    for i in range(n):
        s = s+convTimesSum3DtoZ2(m,ks[i],o[i])
    return s

def softmax(a):
    s = array(a)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            s[i,j]=exp(a[i,j])
    s = s/matmul(np.ones((1,a.shape[0])),matmul(s,np.ones((s.shape[1],1))))
    return s

a = matlib.arange(18).reshape(2,3,3)
b = matlib.arange(16).reshape(2,2,2,2)
c = np.random.randint(0,10,(2,3,3))
d = np.random.random((2,2,2,2))
a[:,1:3,1:3]=matlib.arange(a.shape[0]*4).reshape(a.shape[0],2,2)
#print(a)
#print(b*d)
#os.system("pause")

R = 1
K = 1
N=1
anws1 = np.random.randint(0,1,(1,R,R))
anws2 = np.random.randint(1,2,(1,R,R))
print(anws1)
print(anws2)

x = np.random.randint(0,2,(1,R,R))
w1 = np.random.randint(0,2,(1,x.shape[0],K,K))#*100
#w1 = w1*np.random.randint(-1,2,w1.shape)
b1 = np.random.randint(-1,2,(w1.shape[0],1,1))#*0-50
w2 = np.random.randint(0,2,(1,w1.shape[0],K,K))#*100
b2 = np.random.randint(-1,2,(w2.shape[0],1,1))#*0-50
                        
r_w3 = w2.shape[0]*(int)((((int)(x.shape[1]-w1.shape[2]+2)/2)-w2.shape[2]+2)/2)*(int)((((int)(x.shape[2]-w1.shape[3]+2)/2)-w2.shape[3]+2)/2)
print(r_w3)
w3 = np.random.random((r_w3*1+0*R*R,R*R*R*N))*(10**-(1*R))
b3 = np.random.random((1,w3.shape[1]))
w4 = np.random.random((w3.shape[1],R*R*R*N))*(10**-(1*R))
b4 = np.random.random((1,w4.shape[1]))
w5 = np.random.random((w4.shape[1],2))*(10**-(1*R))
b5 = np.random.random((1,w5.shape[1]))

def forword(x,w1,b1,w2,b2,w3,b3,w4,b4,w5,b5):
    #****forward convolution
    t = conv2D(x[0],w1[0,0])
    x1 = b1*np.ones([w1.shape[0],t.shape[0],t.shape[1]])
    x_1 = np.zeros([w1.shape[0],(int)((t.shape[0]+1)/2),(int)((t.shape[1]+1)/2)])
    for i in range(w1.shape[0]):
        x1[i] = x1[i] + conv3D(x,w1[i])
        x_1[i] = pool2D(x1[i])
    x_1_2 = 1/(1+exp(-x_1))

    t = conv2D(x_1[0],w2[0,0])
    x2 = b2*np.ones([w2.shape[0],t.shape[0],t.shape[1]])
    x_2 = np.zeros([w2.shape[0],(int)((t.shape[0]+1)/2),(int)((t.shape[1]+1)/2)])
    for i in range(w2.shape[0]):
        x2[i] = x2[i] + conv3D(x_1_2,w2[i])
        x_2[i] = pool2D(x2[i])
    x_2_shape = x_2.shape
    x_2 = x_2.reshape((1,x_2.shape[0]*x_2.shape[1]*x_2.shape[2]))
    x_2_2 = 1/(1+exp(-x_2))
    #x_2 = x.reshape((1,R*R))

    #****forward full neural
    x3 = 1/(1+exp(-1*(matmul(x_2_2,w3)+b3)))
    x4 = 1/(1+exp(-1*(matmul(x3,w4)+b4)))
    x5 = 1/(1+exp(-1*(matmul(x4,w5)+b5)))
    return x1,x_1,x_1_2,x2,x_2,x_2_2,x_2_shape,x3,x4,x5
con = 1
L = 1000000
for interior in range(L):
    y = np.array((1,2))
    x = np.array((5,5))
    if(np.random.randint(0,2)==1):
        x = anws1
        y = np.array([[0,1]])
    else:
        x = anws2
        y = np.array([[1,0]])  
        
    x1,x_1,x_1_2,x2,x_2,x_2_2,x_2_shape,x3,x4,x5 = forword(x,w1,b1,w2,b2,w3,b3,w4,b4,w5,b5)
    #y = np.ones((1,w5.shape[1]))
    C = 1/2*(x5-y)**2
    #print("\nx:\n",x,"\nw1:\n",w1,"\nb1:\n",b1,"\nx1:\n",x1,"\nw2:\n",w2,"\nb2:\n",b2,"\nx2:\n",x2,"\nx_2:\n",x_2)
    #print("\nw3:\n",w3,"\nb3:\n",b3,"\nx3:\n",x3,"\nw4:\n",w4,"\nb4:\n",b4,"\nx4:\n",x4,"\nw5:\n",w5,"\nb5:",b5,"\nx5:\n",x5,)

    if(interior < L/10):
        con = 1000
    elif(interior < L/10*3):
        con = 100
    else:
        con = 10
    
    dCdx5 = (softmax(x5)-y)*con
    dCdb5 = dCdx5*x5*(1-x5)
    dCdw5 = matmul(x4.T,dCdb5)
    dCdb4 = matmul(dCdb5,w5.T)*x4*(1-x4)
    dCdw4 = matmul(x3.T,dCdb4)
    dCdb3 = matmul(dCdb4,w4.T)*x3*(1-x3)
    dCdw3 = matmul(x_2.T,dCdb3)
    if(interior%(L/10)==0):
        print("\nC:\n",C,interior)
        #print("\nw3:\n",w3,"\nb3:\n",b3,"\nx3:\n",x3,"\nw4:\n",w4,"\nb4:\n",b4,"\nx4:\n",x4,"\nw5:\n",w5,"\nb5:",b5,"\nx5:\n",x5,)
        #print("\ndCdb5:\n",dCdb5,"\ndCdw5:\n",dCdw5,"\ndCdb4:\n",dCdb4,"\ndCdw4:\n",dCdw4,"\ndCdb3:\n",dCdb3,"\ndCdw3:\n",dCdw3)
        #os.system("pause")

    dCdx_2 = (matmul(dCdb3,w3.T)*x_2_2*(1-x_2_2)).reshape(x_2_shape)
    dCdb2 = sum3DtoZ(dCdx_2)    #*1/4          *1/(x_2_shape[1]*x_2_shape[2])
    dCdw2 = multiConvSum(x_1,w2,extend3DonXY(dCdx_2,x2))

    #print(dCdx_2.shape,x_1_2.shape)
    dCdx_1 = multiConvSum2(x_1,w2,extend3DonXY(dCdx_2,x2))*x_1_2*(1-x_1_2)
    dCdb1 = sum3DtoZ(dCdx_1)
    dCdb1 = dCdb1               #*1/4                     *1/(x_1.shape[1]*x_1.shape[2])
    #dCdw1 = multiConvSum(x.reshape((1,x.shape[0],x.shape[1])),np.ones((w1.shape[0],1,w1.shape[1],w1.shape[2])),dCdx_1).reshape(w1.shape)
    dCdw1 = multiConvSum(x,w1,extend3DonXY(dCdx_1,x1))
    dCdw1 = dCdw1                         #*1/(x.shape[0]*x.shape[1])
    #if(interior%500==0):
        #print("\ndCdb5:\n",dCdb5,"\ndCdw5:\n",dCdw5,"\ndCdb4:\n",dCdb4,"\ndCdw4:\n",dCdw4,"\ndCdb3:\n",dCdb3,"\ndCdw3:\n",dCdw3)
        #print("\ndCdb2:\n",dCdb2,"\ndCdw2:\n",dCdw2,"\ndCdb1:\n",dCdb1,"\ndCdw1:\n",dCdw1)
        #print(dCdx_2.shape,w2.shape,x1.shape,(x1.shape[0],x1.shape[1]-w2.shape[2]+1,x1.shape[2]-w2.shape[3]+1))
        #os.system("pause")
    
    rate = 1
    conv = 1
    full = 0
    w1 = w1-rate*dCdw1*conv
    b1 = b1-rate*dCdb1*conv
    w2 = w2-rate*dCdw2*conv
    b2 = b2-rate*dCdb2*conv
    w3 = w3-rate*dCdw3*full
    b3 = b3-rate*dCdb3*full
    w4 = w4-rate*dCdw4*full
    b4 = b4-rate*dCdb4*full
    w5 = w5-rate*dCdw5*full
    b5 = b5-rate*dCdb5*full

c = 0
for interior in range(20):
    a = np.array((1,2))
    x = np.array((R,R))
    if(np.random.randint(0,2)==1):
        x = anws1
        a = np.array([[0,1]])
    else:
        x = anws2
        a = np.array([[1,0]])
    x1,x_1,x_1_2,x2,x_2,x_2_2,x_2_shape,x3,x4,x5 = forword(x,w1,b1,w2,b2,w3,b3,w4,b4,w5,b5)
    a = abs(x5-a)
    print(a,a[0,0]<0.5,a[0,1]<0.5,x5,softmax(x5))
    if(a[0,0]<0.5 and a[0,1]<0.5):
        c = c+1
print(c)
print("\nw1:\n",w1,"\nb1:\n",b1,"\nw2:\n",w2,"\nb2:\n",b2,)
print("\nw3:\n",w3,"\nb3:\n",b3,"\nw4:\n",w4,"\nb4:\n",b4,"\nw5:\n",w5,"\nb5:",b5)






