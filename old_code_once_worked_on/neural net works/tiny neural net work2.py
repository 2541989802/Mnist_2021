from numpy import array, exp, dot, matmul
import random

random.seed(5)

In = array([[2]])
w1 = array([[1,-1,2,3,5]])
b1 = array([[1,3,6,2,-9]])
w2 = array([[1,4,3,0,3],[1,0,5,6,4],[3,4,8,7,1],[3,-2,1,-5,3,],[-9,6,-3,3,4]])
b2 = array([[1,7,8,-9,-1]])
w3 = array([[1,0,5,6,4],[1,4,3,0,3],[3,-2,1,-5,3,],[-9,6,-3,3,4],[-3,3,2,1,0]])
b3 = array([[1,7,8,-9,-1]])
w4 = array([[1],[8],[10],[-2],[-3]])

for i in range(100000):
    In[[0]] = 10*random.random()
    
    x1 = 1/(1+exp(-In[0]*w1+b1))
    
    x2 = 1/(1+exp(-1*matmul(x1,w2)+b2))
    
    x3 = 1/(1+exp(-1*matmul(x2,w3)+b3))

    Out = matmul(x3,w4)
    
    if(In[[0]] < 3):
        y = 0
    elif(In[[0]] <6):
        y = 5
    else:
        y = 10
    
    C = 1/2*(Out[0]- y)**2

    dCdO = 10*(Out-y)

    dCdw4 = matmul(x3.T,dCdO)
    dCdb3 = matmul(dCdO, w4.T)*x3*(1-x3)
    dCdw3 = matmul(x2.T,dCdb3)
    dCdb2 = matmul(dCdb3, w3.T)*x2*(1-x2)
    dCdw2 = matmul(x1.T,dCdb2)
    dCdb1 = matmul(dCdb2, w2.T)*x1*(1-x1)
    dCdw1 = matmul(In.T,dCdb1)

    w4 = w4-0.05*dCdw4
    b3 = b3-0.05*dCdb3
    w3 = w3-0.05*dCdw3
    b2 = b2-0.05*dCdb2
    w2 = w2-0.05*dCdw2
    b1 = b1-0.05*dCdb1
    w1 = w1-0.05*dCdw1

    #print(dCdw3,dCdb2,dCdw2,dCdb1,dCdw1)
    #print("*****************")
    if( i%10000 ==0):print(C)

c = 0
for i in range(100):
    
    In[[0]] = 10*random.random()

    x1 = 1/(1+exp(-In[0]*w1+b1))
    
    x2 = 1/(1+exp(-1*matmul(x1,w2)+b2))
    
    x3 = 1/(1+exp(-1*matmul(x2,w3)+b3))

    Out = matmul(x3,w4)

    print(In,Out)

    if(In[[0]] >=0 and In[[0]] <3 and (Out>= -2.5 and Out < 2.5)):
        c +=1
    elif(In[[0]] >=3 and In[[0]] <6 and (Out>= 2.5 and Out < 7.5)):
        c +=1
    elif(In[[0]] >=6 and In[[0]] <= 10 and (Out>= 7.5 and Out < 12.5)):
        c +=1
print("\nw1:\n",w1,"\nb1:\n",b1,"\nw2:\n",w2,"\nb2:\n",b2,"\nw3:\n",w3,"\nb3:\n",b3,"\nw4:\n",w4,)

print(c)
