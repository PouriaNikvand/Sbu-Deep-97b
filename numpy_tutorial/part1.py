# data structure review
# integer and float
import pickle

x=3
# print(float(x))




#_______________________________________________________________________________________________________________________
# boolean
t=True
f=False
# and or not
# print(t is f)


#_______________________________________________________________________________________________________________________
# string    captalize
hello="hello"
world="world"


# print(hello)





#_______________________________________________________________________________________________________________________
# dictionary    pickle


p=open("r.pickle","rb")
t=pickle.load(p)
p.close()

print(t)

# for key,value in t.items():
#     print(value)


# print(t)
s=set()