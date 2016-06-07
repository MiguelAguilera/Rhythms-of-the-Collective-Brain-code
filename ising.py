#!/usr/bin/python

import numpy as np
from itertools import combinations

def get_PDF(h,J,size):	#Get probability density function of ising model with parameters h, J
	
	P=np.zeros(2**size)
	for n in range(2**size):
		s=bitfield(n,size)*2-1
		P[n]=np.exp(np.dot(s,h) + np.dot(np.dot(s,J),s))
	
	P/=np.sum(P)
	
	return P

def random_ising(size):
	h=np.random.randn(size)
	J=np.zeros((size,size))
	for i in np.arange(size):
		for j in np.arange(i+1,size):
			J[i,j]=np.random.randn(1)
	return h,J

def independent_model(Pi):
#	size=int(np.log2(len(Pr)))
	size=Pi.shape[0]
	h=np.zeros((size))
	for i in range(size):
		Pri=subPDF(Pr,[i])
		h[i]=0.5*(np.log(Pri[1]/Pri[0]))
	J=np.zeros((size,size))
	return h,J
	
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
#	for i in range(size):
#		for j in np.arange(i+1,size):
#			C[i,j]-=m[i]*m[j]
	return m,C

def MCsamples(h,J,samples):
	size=len(h)
	s = np.random.randint(0,2,size)*2-1
	# Main simulation loop:
	P={}
	for t in range(samples):
		s=MetropolisStep(s,h,J,size)
		n=bool2int((s+1)/2)
		P[n]=np.exp(np.dot(s,h) + np.dot(np.dot(s,J),s))
	return P
		
def observablesMC(ns,P,size):	#Get mean and correlations from Monte Carlo simulation of ising model

	m=np.zeros((size))
	C=np.zeros((size,size))
	
	for ind,n in enumerate(ns):
		s=bitfield(n,size)*2-1
		for i in range(size):
			m[i]+=P[ind]*s[i]
			for j in np.arange(i+1,size):
				C[i,j]+=P[ind]*s[i]*s[j]
#	for i in range(size):
#		for j in np.arange(i+1,size):
#			C[i,j]-=m[i]*m[j]
	return m,C

def inverse_exact(m1,C1,h,J,error):

	size=len(h)
	u=0.1
	count=0
	
	P=get_PDF(h,J,size)
	(m,C)=observables(P,size)
	fit = np.mean((m-m1)**2)
	fit = 0.5*fit + 0.5*np.mean((C-C1)**2)
	fmin=fit

	while fit>error:


		P=get_PDF(h,J,size)
		(m,C)=observables(P,size)

		dh=u*(m1-m)
		h+=dh
		dJ=u*(C1-C)
		J+=dJ
		fit = np.mean((m-m1)**2)
		fit = 0.5*fit + 0.5*np.mean((C-C1)**2)
		
#		if count%10==0:
#			print size,count,fit
		count+=1
	return h,J,fit
	
def inverseMC(m1,C1,h,J,error):

	size=len(h)
	u=0.1
	samples=100
	nT=40

	fit=1E10
	fmin=fit
	fitcount=0

	Ps = MCsamples(h,J,samples)
	PS=[Ps]*nT

	count=0
	while fit>error:
		del PS[0]
		PS+=[MCsamples(h,J,samples)]
		Ps=PS[0]
		for i in np.arange(1,nT):
			Ps.update(PS[i])

		ns=Ps.keys()
		Pmc=Ps.values()
		Pmc/=np.sum(Pmc)
		m,C=observablesMC(ns,Pmc,size)
	
		dh=u*(m1-m)
		h+=dh
		dJ=u*(C1-C)
		J+=dJ
	
		fit = 0.5*( np.mean((m-m1)**2) + 0.5*np.mean((C-C1)**2) )
		if fit/fmin<1:
			fmin=fit
			fitcount=0
		else:
			fitcount+=1
			if fitcount>nT:
				if len(Ps)/2.0**size<1:
					samples+=samples/2
				fitcount=0
#		if count%10==0:
#			print size,count,len(Ps)/2.0**size,samples,fit
		count+=1
	
	return h,J,fit
	
def MetropolisStep(s,h,J,size,i=None):
	if i is None:
		i = np.random.randint(size)
	eDiff = deltaE(i,s,h,J)
	if eDiff <= 0 or np.random.rand(1) < np.exp(-eDiff):    # Metropolis!
		s[i] = -s[i]
	return s

def GlauberStep(s,h,J,size,i=None):
	if i is None:
		i = np.random.randint(size)
	eDiff = deltaE(i,s,h,J)
	if np.random.rand(1) < 1.0/(1.0+np.exp(eDiff)):    # Glauber!
		s[i] = -s[i]
	return s
	
def deltaE(i,s,h,J):
	return 2*(s[i]*h[i] + np.sum(s[i]*(J[i,:]*s)+s[i]*(J[:,i]*s)))
 
def metastable_states(P):
	size=int(np.log2(len(P)))
	ms=[]
	Pms=[]
	for n in range(len(P)):
		m=1
		s=bitfield(n,size)
		for i in range(size):
			s1=s.copy()
			s1[i]=1-s1[i]
			n1=bool2int(s1)
			if P[n]<P[n1]:
				m=0
				break
		if m==1:
			ms+=[n]
			Pms+=[P[n]]
	return ms,Pms
	
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
