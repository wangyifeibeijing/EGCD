import numpy as np






features_in=[[1,3,5,2],
             [3,1,4,7],
             [2,3,6,6],
             [5,5,7,7],]
x=to_ones(features_in)
print(x)
i=1
x1=x[i,0]
x2=x[i,1]
x3=x[i,2]
x4=x[i,3]
print(x1,x2,x3)
print(x1*x1+x2*x2+x3*x3+x4*x4)