#broadcasting
import numpy as np
x=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])#4*3
v=np.array([1,0,1]) #1*3
# print(x+v)
y=np.empty_like(x)
# for i in range(4):
#     y[i,:]=x[i,:]+v
# print(y)
#
# w=np.tile(v,(4,1))
# print(w)
# print(x+w)
#
print(v+x)

















































































