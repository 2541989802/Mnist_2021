import numpy as np
import numpy
import cupy, struct
from numpy import array, ones, zeros, matmul, exp, random
import os, time
from PIL import Image

def tuple_replace(tu,n,i):
    l = list(tu)
    l[i] = n
    return tuple(l)

def tuple_add(t1,t2):
    if(len(t1)!=len(t2)):
        print("warning: tuple_add, tuple1 has different length with tuple2")
        os.system("pause")
    l1 = list(t1)
    l2 = list(t2)
    for i in range(len(t1)):
        l1[i] = l1[i] + l2[i]
    return tuple(l1)

def expand2D(matrix,kernal,restrain_r=-1,restrain_c=-1): #simple expand and restrain
    m_r = matrix.shape[0]
    m_c = matrix.shape[1]
    k_r = kernal.shape[0]
    k_c = kernal.shape[1]
    r_r = m_r*k_r
    r_c = m_c*k_c
    if((restrain_r!=-1 and r_r<restrain_r) or (restrain_c!=-1 and r_c<restrain_c)):
        print("warning: expand2D, restrain is/are greater than largest possible result")
        os.system("pause")
    if(restrain_r!=-1):
        r_r = restrain_r
    if(restrain_c!=-1):
        r_c = restrain_c
    _m_r = m_r*k_r
    _m_c = m_c*k_c
    _matrix = zeros((_m_r,_m_c))
    result = zeros((r_r,r_c))
    for i in range(_m_r):
        if(restrain_r!=-1 and i>restrain_r):
            break
        for j in range(_m_c):
            if(restrain_c!=-1 and j>restrain_c):
                break
            if(i%k_r==0 and j%k_c==0):
                _matrix[i:i+k_r,j:j+k_r]=matrix[round(i/k_r),round(j/k_c)]*kernal
    result = _matrix[0:r_r,0:r_c]
    return result

def expand3D(box,kernal,restrain_r=-1,restrain_c=-1):
    b_z = box.shape[0]
    b_r = box.shape[1]
    b_c = box.shape[2]
    k_z = kernal.shape[0]
    k_r = kernal.shape[1]
    k_c = kernal.shape[2]
    r_r = b_r*k_r
    r_c = b_c*k_c
    if(b_z!=k_z):
        print("warning: expand3D, box have different channel with kernal")
        os.system("pause")
    if(restrain_r!=-1):
        r_r = restrain_r
    if(restrain_c!=-1):
        r_c = restrain_c
    result = zeros((b_z,r_r,r_c))
    for i in range(b_z):
        result[i] = expand2D(box[i],kernal[i],restrain_r,restrain_c)
    return result

def conv2D_step(matrix,kernal,step_r=1,step_c=1,fill=True): #simple filling
    m_r = matrix.shape[0]
    m_c = matrix.shape[1]
    k_r = kernal.shape[0]
    k_c = kernal.shape[1]
    r_r = (int)(m_r/step_r)
    r_c = (int)(m_c/step_c)
    if(m_r<k_r or m_c<k_c):
        print("warning: conv2D_step, matrix smaller than kernal")
        os.system("pause")
    if(fill):
        r_r = r_r if(m_r%step_r==0) else r_r + 1
        r_c = r_c if(m_c%step_c==0) else r_c + 1
    else:
        r_r = (int)((m_r-k_r+1)/step_r) if((m_r-k_r+1)>=step_r) else 1
        r_c = (int)((m_c-k_c+1)/step_c) if((m_c-k_c+1)>=step_c) else 1
    _matrix = zeros((round((r_r-1)*step_r+k_r),round((r_c-1)*step_c+k_c)))
    result = zeros((r_r,r_c))
    if(fill):
        _matrix[0:m_r,0:m_c] = matrix
    else:
        _matrix = matrix[0:_matrix.shape[0],0:_matrix.shape[1]]
    for i in range(_matrix.shape[0]-k_r+1):
        for j in range(_matrix.shape[1]-k_c+1):
            if(i%step_r==0 and j%step_c==0):
                result[round(i/step_r),round(j/step_c)] = np.sum(_matrix[i:i+k_r,j:j+k_c]*kernal)
    return result

def conv3D_step(box,kernal,step_r=1,step_c=1,fill=True):
    b_z = box.shape[0]
    b_r = box.shape[1]
    b_c = box.shape[2]
    k_z = kernal.shape[0]
    k_r = kernal.shape[1]
    k_c = kernal.shape[2]
    r_r = (int)(b_r/step_r)
    r_c = (int)(b_c/step_c)
    if(b_z!=k_z):
        print("warning: conv3D_step, box have different channel with kernal")
        os.system("pause")
    if(fill):
        r_r = r_r if(b_r%step_r==0) else r_r + 1
        r_c = r_c if(b_c%step_c==0) else r_c + 1
    result = zeros((b_z,r_r,r_c))
    for i in range(b_z):
        result[i] = conv2D_step(box[i],kernal[i],step_r,step_c,fill)
    return result

def conv2D(matrix,kernal):
    m_r = matrix.shape[0]
    m_c = matrix.shape[1]
    k_r = kernal.shape[0]
    k_c = kernal.shape[1]
    if(m_r<k_r or m_c<k_c):
        print("warning: conv2D, matrix smaller than kernal")
        os.system("pause")
    result = zeros((m_r-k_r+1,m_c-k_c+1))
    for i in range(k_r):
        for j in range(k_c):
            result = result + matrix[i:i+result.shape[0],j:j+result.shape[1]]*kernal[i,j]
    return result

def conv3D(box,kernal):
    b_z = box.shape[0]
    b_r = box.shape[1]
    b_c = box.shape[2]
    k_z = kernal.shape[0]
    k_r = kernal.shape[1]
    k_c = kernal.shape[2]
    if(b_r<k_r or b_r<k_c):
        print("warning: conv3D, matrix smaller than kernal")
        os.system("pause")
    if(b_z!=k_z):
        print("warning: conv3D, box have different channel with kernal")
        os.system("pause")
    result = zeros((k_z,b_r-k_r+1,b_c-k_c+1))
    for j in range(k_r):
        for k in range(k_c):
            result = result + box[:,j:j+result.shape[1],k:k+result.shape[2]]*kernal[:,j:j+1,k:k+1]
    result = sum(result)
    return result

def convolute(pic,kernals):
    pic_z = pic.shape[0]
    pic_r = pic.shape[1]
    pic_c = pic.shape[2]
    k_n = kernals.shape[0]
    k_z = kernals.shape[1]
    k_r = kernals.shape[2]
    k_c = kernals.shape[3]
    if(pic_r<k_r or pic_r<k_c):
        print("warning: conv3D, matrix smaller than kernal")
        os.system("pause")
    if(pic_z!=k_z):
        print("warning: conv3D, box have different channel with kernal")
        os.system("pause")
    result = zeros((k_n,k_z,pic_r-k_r+1,pic_c-k_c+1))
    for j in range(k_r):
        for k in range(k_c):
            result = result + pic[:,j:j+result.shape[2],k:k+result.shape[3]]*kernals[:,:,j:j+1,k:k+1]
    result = np.sum(result,axis=1)
    return result

def dw_conv2D(db,pic,kernal):
    p_r = pic.shape[0]
    p_c = pic.shape[1]
    k_r = kernal.shape[0]
    k_c = kernal.shape[1]
    db_r = db.shape[0]
    db_c = db.shape[1]
    if(tuple_add(db.shape,kernal.shape)!=tuple_add(pic.shape,(1,1))):
        print("warning: dw_conv2D, picture doesn't has expect size")
        os.system("pause")
    if(p_r<k_r or p_c<k_c):
        print("warning: dw_conv2D, picture smaller than kernal")
        os.system("pause")
    result = zeros((k_r,k_c))
    for i in range(k_r):
        for j in range(k_c):
            result[i,j] = sum(sum(pic[i:i+db_r,j:j+db_c]*db))
    return result

def dw_conv3D(db,pic,kernal):
    p_z = pic.shape[0]
    p_r = pic.shape[1]
    p_c = pic.shape[2]
    k_z = kernal.shape[0]
    k_r = kernal.shape[1]
    k_c = kernal.shape[2]
    db_r = db.shape[0]
    db_c = db.shape[1]
    if(tuple_add(db.shape,kernal[0].shape)!=tuple_add(pic[0].shape,(1,1))):
        print("warning: dw_conv3D, picture doesn't has expect size")
        os.system("pause")
    if(p_r<k_r or p_c<k_c):
        print("warning: dw_conv3D, picture smaller than kernal")
        os.system("pause")
    if(p_z!=k_z):
        print("warning: dw_conv3D, picture has different channel with a kernal")
        os.system("pause")
    #_db = db*ones((3,1,1))
    result = zeros((k_z,k_r,k_c))
    for i in range(k_r):
        for j in range(k_c):
            result[:,i:i+1,j:j+1] = np.sum(pic[:,i:i+db_r,j:j+db_c]*db,axis=(1,2),keepdims=True)
    return result

def dw_convolute(db,pic,kernals):
    p_z = pic.shape[0]
    p_r = pic.shape[1]
    p_c = pic.shape[2]
    k_n = kernals.shape[0]
    k_z = kernals.shape[1]
    k_r = kernals.shape[2]
    k_c = kernals.shape[3]
    db_n = db.shape[0]
    db_r = db.shape[1]
    db_c = db.shape[2]
    if(db_n!=k_n):
        print("warning: dw_convolute, gradient of b has different channel with the number of kernal")
        os.system("pause")
    result = zeros((k_n,k_z,k_r,k_c))
    for i in range(k_n):
        result[i] = dw_conv3D(db[i],pic,kernals[i])
    return result

def dx_conv2D(db,pic,kernal):
    p_r = pic.shape[0]
    p_c = pic.shape[1]
    k_r = kernal.shape[0]
    k_c = kernal.shape[1]
    db_r = db.shape[0]
    db_c = db.shape[1]
    if(tuple_add(db.shape,kernal.shape)!=tuple_add(pic.shape,(1,1))):
        print("warning: dx_conv2D, picture doesn't has expect size")
        os.system("pause")
    if(p_r<k_r or p_c<k_c):
        print("warning: dx_conv2D, picture smaller than kernal")
        os.system("pause")
    result = zeros((p_r,p_c))
    for i in range(k_r):
        for j in range(k_c):
            result[i:i+db_r,j:j+db_c] = result[i:i+db_r,j:j+db_c] + kernal[i,j]*db
    return result

def dx_conv3D(db,pic,kernal):
    p_z = pic.shape[0]
    p_r = pic.shape[1]
    p_c = pic.shape[2]
    k_z = kernal.shape[0]
    k_r = kernal.shape[1]
    k_c = kernal.shape[2]
    db_r = db.shape[0]
    db_c = db.shape[1]
    if(tuple_add(db.shape,kernal[0].shape)!=tuple_add(pic[0].shape,(1,1))):
        print("warning: dx_conv3D, picture doesn't has expect size")
        os.system("pause")
    if(p_r<k_r or p_c<k_c):
        print("warning: dx_conv3D, picture smaller than kernal")
        os.system("pause")
    if(p_z!=k_z):
        print("warning: dx_conv3D, picture has different channel with a kernal")
        os.system("pause")
    result = zeros((p_z,p_r,p_c))
    for i in range(k_r):
        for j in range(k_c):
            result[:,i:i+db_r,j:j+db_c] = result[:,i:i+db_r,j:j+db_c] + kernal[:,i:i+1,j:j+1]*db
    return result

def dx_convolute(db,pic,kernals):
    p_z = pic.shape[0]
    p_r = pic.shape[1]
    p_c = pic.shape[2]
    k_n = kernals.shape[0]
    k_z = kernals.shape[1]
    k_r = kernals.shape[2]
    k_c = kernals.shape[3]
    db_n = db.shape[0]
    db_r = db.shape[1]
    db_c = db.shape[2]
    if(db_n!=k_n):
        print("warning: dx_convolute, gradient of b has different channel with the number of kernal")
        os.system("pause")
    result = zeros((p_z,p_r,p_c))
    for i in range(k_n):
        result = result + dx_conv3D(db[i],pic,kernals[i])
    return result

def softmax(y): #for one set output
    result = array(y)
    result = exp(y)
    d = np.sum(result)
    result = result/d
    return result

def m_h(_x,func): #matrix help
    shape = _x.shape
    x = np.reshape(_x,(1,-1))
    for i in range(x.shape[1]):
        x[0,i] = func(x[0,i])
    x = x.reshape(shape)
    return x

def d_sigmoid(_x):
    x = _x*(1-_x)
    return x

def sigmoid(x):
    result = array(x)
    result = 1/(1+exp(-x))
    return result

def relu_h(_x):
    x = _x
    if(x>0):
        x = 1*x
    else:
        x = 0.01*x
    return x

def relu(_x):
    return m_h(_x,relu_h)

def d_relu_h(_x):
    x = _x
    if(x>0):
        x = 1
    else:
        x = 0.01
    return x

def d_relu(_x):
    return m_h(_x,d_relu_h)

def trainset_read_h(pic,i,j,offset): # 0 as 1 include i, not include j
    start=16+i*28*28
    if(pic.shape[2]!=28 and pic.shape[3]!=28):
        print("warning, pic size is different from (28,28)")
        os.system("pause")
    with open("variables/train-images.idx3-ubyte","rb") as f:
        f.seek(start,0)
        for l in range(j-i):
            for k in range(28):
                pic[i+l-offset,0][k]=array((struct.unpack('!BBBBBBBBBBBBBBBBBBBBBBBBBBBB',f.read(28))))

def trainlabel_read_h(ys,i,j,offset): # 0 as 1 include i, not include j
    start=8+i
    with open("variables/train-labels.idx1-ubyte","rb") as f:
        f.seek(start,0)
        for l in range(j-i):
                ys[i+l-offset]=struct.unpack('!B',f.read(1))[0]

def trainset_read_Batch(pic): # 0 as 1 include i, not include j
    start=16
    size = 28*28
    if(pic.shape[2]!=28 and pic.shape[3]!=28):
        print("warning: pic size is different from (28,28)")
        os.system("pause")
    ids = zeros((pic.shape[0]))
    with open("variables/train-images.idx3-ubyte","rb") as f:
        for i in range(pic.shape[0]):
            sample = (int)(random.random()*60000)
            ids[i] = sample
            f.seek(start+sample*size,0)
            for j in range(28):
                pic[i,0][j]=array((struct.unpack('!BBBBBBBBBBBBBBBBBBBBBBBBBBBB',f.read(28))))
    return ids

def trainlabel_read_Batch(ys,ids): # 0 as 1 include i, not include j
    start=8
    if(ys.shape[0]!=ids.shape[0]):
        print("warning: ys has different length than ids")
        os.system("pause")
    with open("variables/train-labels.idx1-ubyte","rb") as f:
        for i in range(ys.shape[0]):
            sample = round(ids[i])
            f.seek(start+sample,0)
            ys[i]=struct.unpack('!B',f.read(1))[0]

def resize_pic():
    path = "variables/picture/original_size/"
    target = "variables/picture/"
    for file in os.listdir(path):
        if(not os.path.isdir(path+file)):
            im = Image.open(path+file)
            im = im.convert("L").resize((28,28))
            im.save(target+file)

def read_pic():
    l = []
    l2 = []
    path = "variables/picture/"
    for file in os.listdir(path):
        if(not os.path.isdir(path+file)):
            im = Image.open(path+file).resize((28,28))
            l.append(255-array(im))
            l2.append(int(file[0]))
    m = zeros((len(l),1,28,28))
    label = zeros((len(l),10))
    for i in range(len(l)):
        m[i,0] = l[i]
        label[i,l2[i]] = 1
    return m,label,l2







