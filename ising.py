#!/usr/bin/python

import numpy as np
from itertools import combinations

def get_ising_PDF(h,J,size):	#Get probability density function of ising model with parameters h, J
	
	P=np.zeros(2**size)
	for n in range(2**size):
		s=bitfield(n,size)*2-1
		P[n]=np.exp(np.dot(s,h) + np.dot(np.dot(s,J),s))
	
	P/=np.sum(P)
	
	return P
	
def bool2int(x):				#Transform bool array into positive integer
    y = 0
    for i,j in enumerate(np.array(x)[::-1]):
        y += j<<i
    return y
    
def bitfield(n,size):			#Transform positive integer into bit array
    x = [int(x) for x in bin(n)[2:]]
    x = [0]*(size-len(x)) + x
    return np.array(x)
	
	
def observables(P,size):		#Get mean and correlations from probability density function

	m=np.zeros((size))
	C=np.zeros((size,size))
	for n in range(len(P)):
		s=bitfield(n,size)*2-1
		for i in range(size):
			m[i]+=P[n]*s[i]
			for j in np.arange(i+1,size):
				C[i,j]+=P[n]*s[i]*s[j]
	return m,C

def observablesMC(h,J,size,samples):	#Get mean and correlations from Monte Carlo simulation of ising model

	s = np.random.randint(0,2,size)*2-1
	m=np.zeros((size))
	C=np.zeros((size,size))
	
	# Main simulation loop:
	for t in range(samples):
		s=MetropolisStep(s,h,J,size)
		
		# Compute means and correlations
		for i in range(size):
			m[i]+=s[i]
			for j in np.arange(i+1,size):
				C[i,j]+=s[i]*s[j]

	m/=float(samples)
	C/=float(samples)
	return m,C
	
def MetropolisStep(s,h,J,size):
	i = np.random.randint(size)
	eDiff = deltaE(i,s,h,J)
	if eDiff <= 0 or np.random.rand(1) < np.exp(-eDiff):    # Metropolis!
		s[i] = -s[i]
	return s
	
def deltaE(i,s,h,J):
	return 2*(s[i]*h[i] + np.sum(s[i]*(J[i,:]*s)+s[i]*(J[:,i]*s)))
	
def observablesMC1(h,J0,size,samp):
	
	J=0.5*(J0+np.transpose(J0))
	
	m=np.zeros(size)
	C=np.zeros((size,size))

	Pc=0
	samp=min(2**size,samp)
	for n in np.random.permutation(range(2**size))[0:samp]:
		
		s=bitfield(n,size)*2-1
		e=np.exp(np.dot(s,h) + np.dot(np.dot(s,J),s))
		Pn=np.exp(e)
		
		Pc+=Pn
		for i in range(size):
			m[i]+=Pn*s[i]
			for j in np.arange(i+1,size):
				C[i,j]+=Pn*s[i]*s[j]
	m/=float(Pc)
	C/=float(Pc)

	return m,C,Pc
 

def subPDF(P,rng):
	subsize=len(rng)
	Ps=np.zeros(2**subsize)
	size=int(np.log2(len(P)))
	for n in range(len(P)):
		s=bitfield(n,size)
		Ps[bool2int(s[rng])]+=P[n]
	return Ps
	
def Entropy(P):
	E=0.0
	for n in range(len(P)):
		if P[n]>0:
			E+=-P[n]*np.log2(P[n])
	return E
	

def MI(Pxy, rngx, rngy):
	size=int(np.log2(len(Pxy)))
	Px=subPDF(Pxy,rngx)
	Py=subPDF(Pxy,rngy)
	I=0.0
	for n in range(len(Pxy)):
		s=bitfield(n,size)
		if Pxy[n]>0:
			I+=Pxy[n]*np.log(Pxy[n]/(Px[bool2int(s[rngx])]*Py[bool2int(s[rngy])]))
	return I
	
def TSE(P):
	size=int(np.log2(len(P)))
	C=0
	for npart in np.arange(1,0.5+size/2.0).astype(int):	
		bipartitions = list(combinations(range(size),npart))
		for bp in bipartitions:
			bp1=list(bp)
			bp2=list(set(range(size)) - set(bp))
			C+=MI(P, bp1, bp2)/float(len(bipartitions))
	return C
	
def KL(P,Q):
	D=0
	for i in range(len(P)):
		D+=P[i]*np.log(P[i]/Q[i])
	return D
    
def JSD(P,Q):
	return 0.5*(KL(P,Q)+KL(Q,P))
