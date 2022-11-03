import numpy as np
import cupy as cp
import threading,time,os,struct
from numpy import array,random,zeros,ones,arange,exp
import matrix2_second_try_tool as tool
import variables as var
from PIL import Image

'''im = Image.open("variables/picture/original_size/1_a.png")
im = im.convert("L").resize((28,28))
im_array =  255-array(im)
im.save("variables/picture/1_a.png")
print(im_array)'''



tool.resize_pic()
a = read_pic()
print(a)
