# numpy and list
# less memory
# faster
# convenient
import numpy as np
import time
import sys


size=1000000
l1=list(range(size))
l2=list(range(size))


A1=np.arange(size)
A2=np.arange(size)


# print(A1+A2)

start=time.time()
r=[x+y for x,y in zip(l1,l2)]
# print((time.time()-start)*1000)

start=time.time()
r=A1+A2
# print((time.time()-start)*1000)

# print(sys.getsizeof(l1)*len(l1))
# print(A1.size*A1.itemsize)



#_______________________________________________________________________________________________________________________
# list  append  pop remove  del pickle  read    write   for
# list_one=list(range(5))
list_two=[6,7,8,9,10]

# print(list_one)
# del list_two[0]
# print(list_two.remove(10))
print(np.asarray(list_two))
