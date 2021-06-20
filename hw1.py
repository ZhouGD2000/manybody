import numpy as np
from numpy import pi
from numpy import linalg
from numpy.core.fromnumeric import reshape, trace
from numpy.lib.function_base import meshgrid
from numpy.lib.nanfunctions import nanquantile
from numpy.lib.polynomial import RankWarning
from numpy.linalg import eig as eig
from numpy.linalg.linalg import norm
from scipy.special import erf
from matplotlib import pyplot as plt

phiH=np.array([[0.3425250914E+01,0.1543289673E+00],[0.6239137298E+00, 0.5353281423E+00],[0.1688554040E+00,0.4446345422E+00]])

phiHe=np.array([[0.6362421394E+01,0.1543289673E+00],[0.1158922999E+01,0.5353281423E+00],[0.3136497915E+00,0.4446345422E+00]])

RH=0
RHe=1.4632



def Solver(N=2,phi=[phiH,phiHe],R=[RH,RHe],Z=[1,2]):
    if N%2!=0:
        raise Exception("No suitable for RHF")
    K=len(phi)
    (C,e)=initial(K)
    #K:number of orbit N:number of electron C:K*K,e:K*1
    err=1E-8
    E=min(e)
    (S,T,V,I)=generate(phi,R,R,Z,K)
    while True:
        p=P(C,N//2)
        G=np.einsum("lk,ijkl",p,I)-1/2*np.einsum("lk,ilkj",p,I)
        F=T+V+G
        (C,e)=update(S,F)
        Enew=min(e)
        print(Enew)
        cnew=C[:,e==Enew]
        Enew=np.trace(np.dot(cnew.conj().T,np.dot(2*(T+V)+G,cnew)))
        print(Enew)
        if abs(Enew-E)<err:
            return (Enew,cnew)
        else:
            E=Enew
            
def density(x,y,c,phi,R,n=2):
    # matrix 
    #r: k*l matrix
    K=len(phi)
    phir=np.zeros((len(x),len(y),K))
    for i in range(K):
        phir[:,:,i]=phi_wavefunction(x,y,phi[i],R[i])
    phimunu=phir.reshape((len(x),len(y),K,1))*phir.reshape((len(x),len(y),1,K))
    #phimur: mu,r matrix phinu nu,r P:mu*nu
    return np.einsum("ij,klji",P(c,n//2),phimunu)

def phi_wavefunction(x,y,phii,Ri):
    (X,Y)=np.meshgrid(x,y)
    amplitude=0
    for coeff in phii:
        amplitude+=np.exp(-coeff[0]*((X-Ri)**2+Y**2))
    return amplitude

def initial(K):
    #cof=np.identity(K)
    #cof=np.zeros((K,K))
    #cof=np.ones((K,K))
    cof=np.array([[0,0],[0,0]])
    E=np.zeros(K)
    return (cof,E)


def generate(phi,R,RI,Z,K):
    Sij=np.zeros((K,K))
    Tij=np.zeros((K,K))
    Vij=np.zeros((K,K))
    Iijkl=np.zeros((K,K,K,K))
    for i in range(K):
        for j in range(K):
            phi1=phi[i]
            phi2=phi[j]
            d1=phi1[:,1]
            d2=phi2[:,1]
            R1=R[i]
            R2=R[j]
            (a,b)=np.meshgrid(phi1[:,0],phi2[:,0])
            sab=s(a,b,R1,R2)
            Sij[i][j]=np.dot(d1.conj(),np.dot(sab,d2))
            Tij[i][j]=np.dot(d1.conj(),np.dot(t(a,b,R1,R2,sab),d2))
            Vij[i][j]=np.dot(d1.conj(),np.dot(v(a,b,R1,R2,sab,Z,RI),d2))
            for k in range(2):
                for l in range(2):
                    phi3=phi[k]
                    phi4=phi[l]
                    d3=phi3[:,1]
                    d4=phi4[:,1]
                    R3=R[k]
                    R4=R[l]
                    (c,d)=np.meshgrid(phi3[:,0],phi4[:,0])
                    scd=s(c,d,R3,R4)
                    Iijkl[i][j][k][l]=np.einsum("ijkl,i,j,k,l",interaction(a,b,c,d,R1,R2,R3,R4,sab,scd),d1.conj(),d2,d3.conj(),d4)

    #print("V",Vij)
    #print("T",Tij)
    #print("I",Iijkl)
    return (Sij,Tij,Vij,Iijkl)

"""
def generate(phi,R,C,RI,Z,K,N):
    Sij=np.zeros((K,K))
    Tij=np.zeros((K,K))
    Vij=np.zeros((K,K))
    Gij=np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            phi1=phi[i]
            phi2=phi[j]
            d1=phi1[:,1]
            d2=phi2[:,1]
            R1=R[i]
            R2=R[j]
            (a,b)=np.meshgrid(phi1[:,0],phi2[:,0])
            sab=s(a,b,R1,R2)
            Sij[i][j]=np.dot(d1.conj(),np.dot(sab,d2))
            Tij[i][j]=np.dot(d1.conj(),np.dot(t(a,b,R1,R2,sab),d2))
            Vij[i][j]=np.dot(d1.conj(),np.dot(v(a,b,R1,R2,sab,Z,RI),d2))
            Idirect=np.zeros((K,K))
            Iexchange=np.zeros((K,K))
            for k in range(2):
                for l in range(2):
                    phi3=phi[k]
                    phi4=phi[l]
                    d3=phi3[:,1]
                    d4=phi4[:,1]
                    R3=R[k]
                    R4=R[l]
                    (c,d)=np.meshgrid(phi3[:,0],phi4[:,0])
                    Idirect[k][l]=np.einsum("ijkl,i,j,k,l",interaction(a,b,c,d,R1,R2,R3,R4,sab),d1.conj(),d2,d3.conj(),d4)
                    Iexchange[k][l]=np.einsum("ijkl,i,j,k,l",interaction(a,d,c,b,R1,R4,R3,R2,sab),d1.conj(),d4,d3.conj(),d2)
            Pkl=P(C,N//2)
            Gij[i][j]=np.trace(np.dot(Pkl,Idirect-1/2*Iexchange))
    Fij=Tij+Vij+Gij
    print("V",Vij)
    print("T",Tij)
    return (Sij,Fij,Tij+Vij+Fij)
"""


def P(c,n):
    c_occ=c[:,0:n]
    #K*N
    return 2*np.dot(c_occ,np.transpose(c_occ.conj()))


def s(alpha,beta,R1,R2):
    return (4*alpha*beta/(alpha+beta)**2)**(3/4)*np.exp(-alpha*beta/(alpha+beta)*abs(R1-R2)**2)

def t(alpha,beta,R1,R2,s):
    return (3*alpha*beta/(alpha+beta)-2*((alpha*beta)/(alpha+beta))**2*abs(R1-R2)**2)*s

def v(alpha,beta,R1,R2,s,Z,RI):
    RP=(alpha*R1+beta*R2)/(alpha+beta)
    Nnucle=len(Z)
    vi=0
    for i in range(Nnucle):
        cond=np.zeros_like(RP)
        cond[np.abs(RP-RI[i])<1E-8]=1
        vi=vi-2*Z[i]*np.sqrt((alpha+beta)/pi)*s*cond-Z[i]*erf(np.sqrt(alpha+beta)*abs(RP-RI[i]))/(abs(RP-RI[i])+1E-12)*s*(1-cond)
    return vi

def interaction(alpha,beta,gamma,delta,R1,R2,R3,R4,s1,s2):
    sshape=np.shape(s1)
    s1=np.reshape(s1,(sshape[0],sshape[1],1,1))
    s2=np.reshape(s2,(1,1,sshape[0],sshape[1]))
    a=np.reshape(alpha,(sshape[0],sshape[1],1,1))
    b=np.reshape(beta,(sshape[0],sshape[1],1,1))
    c=np.reshape(gamma,(1,1,sshape[0],sshape[1]))
    d=np.reshape(delta,(1,1,sshape[0],sshape[1]))
    RQ=(c*R3+d*R4)/(c+d)
    RP=(a*R1+b*R2)/(a+b)
    alphat=(a+b)*(c+d)/(a+b+c+d)
    cond=np.zeros_like(RP-RQ)
    cond[np.abs(RP-RQ)<1E-8]=1
    return cond*2*s1*s2*np.sqrt(alphat/np.pi)+(1-cond)*s1*s2*erf(np.sqrt(alphat)*abs(RP-RQ))/(abs(RP-RQ)+1E-12)

def update(S,F):
    (s,U)=eig(S)
    X=np.dot(U,np.diag(s**(-0.5)))
    FF=np.dot(np.dot(X.T.conj(),F),X)
    #print("FF",FF)
    (e,CC)=eig(FF)
    C=np.dot(X,CC)
    order=np.argsort(e)
    e=e[order]
    C=C[:,order]
    return (C,e)


e=[]
e1=[]
x=np.arange(0.5,8,0.1)
for i in x:
    (ee,psi)=Solver(2,[phiH,phiH],[0,i],[1,1])
    print("H2 molecur when R=%f"%i,ee,psi)
    e.append(ee+1/i)
plt.plot(x,e)
print(e)

plt.plot(x,e)
plt.xlabel("distance/Bohr")
plt.ylabel("Total energy/Hatree")
plt.title("Energy curve of H2 molecular")
plt.savefig("H2.png")

L=3
NN=60
(ehe,che)=Solver()
ra=np.arange(-L,L,L/NN)
plt.imshow(density(ra,ra,che,[phiH,phiHe],[RH,RHe]))
plt.scatter([NN,NN+1.4632/L*NN],[NN,NN],s=3)
plt.colorbar()
plt.xticks(np.arange(0,2*NN,NN/5),np.round(np.arange(-L,L,L/5),2))
plt.yticks(np.arange(0,2*NN,NN/5),np.round(np.arange(-L,L,L/5),2))
plt.savefig("densityHeH.png")