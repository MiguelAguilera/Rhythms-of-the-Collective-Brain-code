#!/usr/bin/env python

from ising import ising,bool2int
from ising import Entropy
import numpy as np
import time

size=4
T=2000
error=1E-5

x=ising(size)
x.random_wiring()

m1=np.zeros(size)
C1=np.zeros((size,size))
Pr=np.zeros(2**size)
for t in range(T):
	x.SequentialGlauberStep()
	n=bool2int(x.s)
	Pr[n]+=1
	for i in range(size):
		m1[i]+=x.s[i]/float(T)
		for j in np.arange(i+1,size):
			C1[i,j]+=x.s[i]*x.s[j]/float(T)
for i in range(size):
		for j in np.arange(i+1,size):
			C1[i,j]-=m1[i]*m1[j]
Pr/=np.sum(Pr)	

	
print(m1)
print(C1)




x1=ising(size)
x1.independent_model(m1)
x1.pdf()
P1=x1.P.copy()
x1.observables()

start_time = time.time()
fit=x1.inverse_exact(m1,C1,error)
time1=time.time() - start_time


#start_time = time.time()
#fit=x1.inverse_exact(m1,C1,error,mode='coordinated-descent')
#time2=time.time() - start_time
#print("--- %s seconds ---" % (time1))
#print("--- %s seconds ---" % (time2))
#print fit
S1=Entropy(P1)
Sr=Entropy(Pr)
x1.pdf()
P2=x1.P.copy()
S2=Entropy(P2)
I=S1-Sr
I2=S1-S2

print(S1,S2,Sr)
x.pdf()
#print Entropy(x.P.copy())
#print I2,I
print(I2/I)
print(fit)
