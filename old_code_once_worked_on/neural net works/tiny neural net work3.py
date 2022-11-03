from numpy import array, exp, dot, matmul
import numpy as np
import random, os

random.seed(5)

R=1
a = np.random.randint(0,1,(1,R))
b = np.random.randint(1,2,(1,R))

def relu2D(n):
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if(n[i,j]<0):
                n[i,j] = 0.001*n[i,j]
            else:
                n[i,j] = 0.1*n[i,j]
    return n
                
def d_relu2D(n):
    for i in range(n.shape[0]):
        for j in range(n.shape[1]):
            if(n[i,j]<0):
                n[i,j] = 0.001
            else:
                n[i,j] = 0.1
    return n

print(a,b)


L1 = 0
L2 = 0
L3 = 0
L4 = 1
In = np.random.randint(0,2,(1,R))
w1 = np.random.random((R,1)) if L1==1 else np.eye(In.shape[1],In.shape[1])
b1 = np.random.random((1,w1.shape[1]))*L1
w2 = np.random.random((w1.shape[1],1)) if(L2==1) else np.eye(w1.shape[1],w1.shape[1])
b2 = np.random.random((1,w2.shape[1]))*L2
w3 = np.random.random((w2.shape[1],1)) if(L3==1) else np.eye(w2.shape[1],w2.shape[1])
b3 = np.random.random((1,w3.shape[1]))*L3
w4 = np.random.random((w3.shape[1],1))if(L4==1) else np.ones((w3.shape[1],1))
b4 = np.random.random((1,w4.shape[1]))*L4
print("\nb4:\n",b4,"\nw4:\n",w4,"\nb3:\n",b3,"\nw3:\n",w3,"\nb2:\n",b2,"\nw2:\n",w2,"\nb1:\n",b1,"\nw1:\n",w1)
#os.system("pause")

rate = 0
c = 0
for i in range(100000):
    t = np.random.random()
    if(i%1==0):
        if(t<0.5):
            In = b*t
            y = 0
        else:
            In = b*t
            y = 1

    if(i < 1000):
        c = 0.1
        rate = 10
    else:
        c = 1
        rate = 1
    #x1 = 1/(1+exp(-c*(matmul(In,w1)+b1)))if L1==1 else matmul(In,w1)
    x1 = c*relu2D(matmul(In,w1)+b1)if L1==1 else matmul(In,w1)
    
    #x2 = 1/(1+exp(-c*(matmul(x1,w2)+b2)))if L2==1 else matmul(x1,w2)
    x2 = c*relu2D(matmul(x1,w2)+b2)if L2==1 else matmul(x1,w2)
    
    #x3 = 1/(1+exp(-c*(matmul(x2,w3)+b3)))if L3==1 else matmul(x2,w3)
    x3 = c*relu2D(matmul(x2,w3)+b3)if L3==1 else matmul(x2,w3)

    #Out = 1/(1+exp(-c*(matmul(x3,w4)+b4)))if L4==1 else matmul(x3,w4)
    Out = c*relu2D(matmul(x3,w4)+b4) if L4==1 else matmul(x3,w4)

    
    C = 1/2*(Out[0]- y)**2
    
    dCdO = rate*(Out-y)
    #dCdb4 = dCdO*Out*(1-Out)*c if L4==1 else dCdO
    #dCdw4 = matmul(x3.T,dCdb4)
    #dCdb3 = matmul(dCdb4, w4.T)*x3*(1-x3)*c if L3==1 else matmul(dCdb4, w4.T)
    #dCdw3 = matmul(x2.T,dCdb3)
    #dCdb2 = matmul(dCdb3, w3.T)*x2*(1-x2)*c if L2==1 else matmul(dCdb3, w3.T)
    #dCdw2 = matmul(x1.T,dCdb2)
    #dCdb1 = matmul(dCdb2, w2.T)*x1*(1-x1)*c if L1==1 else matmul(dCdb2, w2.T)
    #dCdw1 = matmul(In.T,dCdb1)
    
    dCdb4 = dCdO*d_relu2D(Out)*c if L4==1 else dCdO
    dCdw4 = matmul(x3.T,dCdb4)
    dCdb3 = matmul(dCdb4, w4.T)*d_relu2D(x3)*c if L3==1 else matmul(dCdb4, w4.T)
    dCdw3 = matmul(x2.T,dCdb3)
    dCdb2 = matmul(dCdb3, w3.T)*d_relu2D(x2)*c if L2==1 else matmul(dCdb3, w3.T)
    dCdw2 = matmul(x1.T,dCdb2)
    dCdb1 = matmul(dCdb2, w2.T)*d_relu2D(x1)*c if L1==1 else matmul(dCdb2, w2.T)
    dCdw1 = matmul(In.T,dCdb1)

    b4 = b4-dCdb4*L4
    w4 = w4-dCdw4*L4
    b3 = b3-dCdb3*L3
    w3 = w3-dCdw3*L3
    b2 = b2-dCdb2*L2
    w2 = w2-dCdw2*L2
    b1 = b1-dCdb1*L1
    w1 = w1-dCdw1*L1

    #print(dCdw3,dCdb2,dCdw2,dCdb1,dCdw1)
    #print("*****************")
    if(i%10000==0):
        print(C,'\t',Out,t,'\t',i)
        #print("\ndCdb4:\n",dCdb4,"\ndCdw4:\n",dCdw4,"\ndCdb3:\n",dCdb3,"\ndCdw3:\n",dCdw3,"\ndCdb2:\n",dCdb2,"\ndCdw2:\n",dCdw2,"\ndCdb1:\n",dCdb1,"\ndCdw1:\n",dCdw1)
        #print("\nb4:\n",b4,"\nw4:\n",w4,"\nb3:\n",b3,"\nw3:\n",w3,"\nb2:\n",b2,"\nw2:\n",w2,"\nb1:\n",b1,"\nw1:\n",w1)
        #print("\nOut:\n",Out,"\nx3:\n",x3,"\nx2:\n",x2,"\nx1:\n",x1,"\nIn:\n",In)
        #os.system("pause")

cou = 0
for i in range(50):
    y = 0
    t = np.random.random()
    if(t<0.5):
        In = b*t
        y = 0
    else:
        In = b*t
        y = 1
        
    #x1 = 1/(1+exp(-c*(matmul(In,w1)+b1)))if L1==1 else matmul(In,w1)
    x1 = c*relu2D(matmul(In,w1)+b1)if L1==1 else matmul(In,w1)
    
    #x2 = 1/(1+exp(-c*(matmul(x1,w2)+b2)))if L2==1 else matmul(x1,w2)
    x2 = c*relu2D(matmul(x1,w2)+b2)if L2==1 else matmul(x1,w2)
    
    #x3 = 1/(1+exp(-c*(matmul(x2,w3)+b3)))if L3==1 else matmul(x2,w3)
    x3 = c*relu2D(matmul(x2,w3)+b3)if L3==1 else matmul(x2,w3)

    #Out = 1/(1+exp(-c*(matmul(x3,w4)+b4)))if L4==1 else matmul(x3,w4)
    Out = c*relu2D(matmul(x3,w4)+b4) if L4==1 else matmul(x3,w4)

    print(Out,In)

    if(abs(Out-y)<0.5):
        cou = cou+1
print(cou)
print("\n************************************************************\n")
print("\nb4:\n",b4,"\nw4:\n",w4,"\nb3:\n",b3,"\nw3:\n",w3,"\nb2:\n",b2,"\nw2:\n",w2,"\nb1:\n",b1,"\nw1:\n",w1)
print("\n************************************************************\n")
