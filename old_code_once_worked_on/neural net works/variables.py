from numpy import array,zeros,ones
import cupy
import numpy
import os,time

numpy.set_printoptions(threshold=numpy.inf)

def StrToNumber(s):
    p = 1
    n = 0
    D = False
    nd = 0.1
    d = 0
    E = False
    ep =1
    e = 0
    S = 0
    for c in s:
        if(c=='-' and not E):
            p = -1
        elif(c=='-' and E):
            ep = -1
        elif(c=='e'):
            E = True
        elif(c=='.'):
            D = True
        elif(c.isdigit()):
            S = (int)(c)-(int)('0')
            if(not E and not D):
                n = n*10 + S
            elif( not E and D):
                d = d+S*nd
                nd=nd*0.1
            elif(E):
                e = e*10+S
    n=p*(n+d)*(10**(ep*e))
    return n

def StrtoMatrix(s):
    i = 0
    d = 0
    time.sleep(1)
    while(not((s[i]).isdigit() or s[i]=='+' or s[i]=='-')):
        if(s[i]=='['):
            d = d+1
        i = i+1
    c = ones(d,int)
    r = d
    D = d
    for j in range(len(s)-i):
        if(s[j+i]==',' and d == r):
            c[d-1]=c[d-1]+1
        elif(s[j+i]==']'):
            if(d==r):
                d = d-1
            r = r-1
        elif(s[j+i]=='['):
            r = r+1
    m = zeros(tuple(c))
    l = ""
    c = zeros(c.shape,int)
    a = 0
    for j in range(len(s)):
        if(s[j]==']'):
            a = a+1
        elif(s[j]=='['):
            r = r+1
        elif(s[j]==','):
            m[tuple(c)]=StrToNumber(l)
            l = ""
            if(a!=0):
                while(a != 0):
                    c[r-1]=0
                    a = a-1
                    r = r-1
            c[r-1]=c[r-1]+1
            
        else:
            l = l + s[j]
    m[tuple(c)]=StrToNumber(l)
    return m

def format(s):
    s = s.replace('\n',',').replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace("[,",'[').replace(",]",']')
    return s

def v_w1():
    with open("variables/v_w1.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_b1():
    with open("variables/v_b1.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_w2():
    with open("variables/v_w2.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_b2():
    with open("variables/v_b2.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_w3():
    with open("variables/v_w3.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_b3():
    with open("variables/v_b3.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_w4():
    with open("variables/v_w4.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_b4():
    with open("variables/v_b4.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_w5():
    with open("variables/v_w5.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def v_b5():
    with open("variables/v_b5.txt") as f:
        s = format(f.read())
        return StrtoMatrix(s)

def save_temp_v(matrix, filename):
    data_str = matrix.__str__()
    filename = "variables/temp/"+filename
    with open(filename,'w') as f:
        f.write(data_str)

def read_temp_v(filename):
    filename = "variables/"+filename
    with open(filename) as f:
        s = format(f.read())
        return StrtoMatrix(s)


with open("variables/v_b1.txt") as f:
    s = format(f.read())
    print(s,'\n')
with open("variables/v_b2.txt") as f:
    s = format(f.read())
    print(s,'\n')
with open("variables/v_b3.txt") as f:
    s = format(f.read())
    print(s,'\n')
with open("variables/v_b4.txt") as f:
    s = format(f.read())
    print(s,'\n')
with open("variables/v_b5.txt") as f:
    s = format(f.read())
    print(s,'\n')











