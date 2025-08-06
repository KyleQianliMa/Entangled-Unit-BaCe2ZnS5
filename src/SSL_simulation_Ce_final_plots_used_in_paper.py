import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic
import sympy as sym
from sympy import symbols, solve,cos,sin,Matrix, Inverse, Number
import pdb
# import pyttsx3
import math
import latex
import cv2
import pandas as pd
plt.rc('legend', fontsize=20)
plt.rc('legend', markerscale=1.5) 
plt.rc('xtick', labelsize=30)    # fontsize of the tick labels
plt.rc('ytick', labelsize=30)
plt.rc('lines', markersize=8)
plt.rc('lines', lw=3)
xlabelsize = 25
ylabelsize = 25
annotatesize = 20
plotlabelsize = 30
a=b=7.91020
c=13.65790
alpha=beta=gamma=np.pi/2
V=a*b*c*np.sqrt(1-np.cos(alpha)**2-np.cos(beta)**2-np.cos(gamma)**2+2*np.cos(alpha)*np.cos(beta)*np.cos(gamma))
a1=[a,0,0]
a2=[0,b,0]
a3=[0,0,c]

b1=2*np.pi*np.cross(a2,a3)/V
b2=2*np.pi*np.cross(a3,a1)/V
b3=2*np.pi*np.cross(a1,a2)/V
pi=np.pi

def rotz(x):
    x=x/180 * np.pi
    Rc=np.matrix([[np.cos(x), -np.sin(x), 0],
                  [np.sin(x),np.cos(x),0],
                  [0, 0, 1]])
    return Rc

def roty(x):
    x=x/180 * np.pi
    Rc=np.matrix([[np.cos(x), 0, np.sin(x)],
                  [0,1,0],
                  [-np.sin(x), 0, np.cos(x)]])
    return Rc

def rotx(x):
    x=x/180 * np.pi
    Rc=np.matrix([[1, 0, 0],
                  [0,np.cos(x),-np.sin(x)],
                  [0,np.sin(x), np.cos(x)]])
    return Rc

def FF(Q):
    A=0.0540
    a=25.0293
    B=0.3101
    b=12.1020
    C=0.6575
    c=4.7223
    dd=-0.0216
    s=Q/(4*np.pi)
    return A*np.exp(-a*s**2)+B*np.exp(-b*s**2)+C*np.exp(-c*s**2)+dd

sx=np.matrix([[0,1/2],
              [1/2,0]])
sy=np.matrix([[0,-1j/2],
              [1j/2,0]])
sz=np.matrix([[1/2,0],
              [0,-1/2]])

idd=np.matrix([[1,0],
              [0,1]])

S1=np.array([np.kron(sx,idd),
             np.kron(sy,idd),
             np.kron(sz,idd)])

S2=np.array([np.kron(idd,sx),
              np.kron(idd,sy),
              np.kron(idd,sz)])

def rotS(rot,S):
    Srx=0
    Sry=0
    Srz=0
    for j in range(3):
        Srx+=rot[0,j]*S[j]
    for j in range(3):
        Sry+=rot[1,j]*S[j]
    for j in range(3):
        Srz+=rot[2,j]*S[j]
    return Srx,Sry,Srz
# S1=rotS(rotx(-90),S1)
# S1=rotS(rotz(45),S1)
# S1=np.round(S1,10)

# S2=rotS(rotx(-90),S2)
# S2=rotS(rotz(45),S2)
# S2=np.round(S2,10) 

###############################
#   Hamiltonian Matrix
###############################

def TT(Jmat):
    ham=0
    for i in range(0,3):
        for j in range(0,3):
            ham=ham+Jmat[i,j]*np.matmul(S1[i],S2[j])
    ev, ef=np.linalg.eigh(ham)
    ef=ef[:,np.argsort(ev)]
    ev=np.sort(ev)
    return ev,ef,ham

def TTM(Jmat,H,g):
    _,_,ham=TT(Jmat)
    muBT=5.7883818012e-2
    hamM=0
    for i in range(0,3):
        hamM=hamM-muBT*(g@H)[i]*(S1[i]+S2[i])
    hamM=hamM+ham
    ev,ef=np.linalg.eigh(hamM)
    ef=ef[:,np.argsort(ev)]
    ev=np.sort(ev)
    return ev,ef,hamM

###############################
#  Kronecker Delta
###############################

def kron_delta(a,b):
    if a==b:
        return 1
    else:
        return 0

############################################
#   Here we difine lattice constants
#   and the interaction between R2-R3
#   and R1-R4. Type in their coordinates
############################################

R1 =[0.33800,   1.16200,   1.00000]
R4 =[0.66200,   0.83800,   1.00000]

R2 =[0.83800,   1.33800,   1.00000]
R3 =[1.16200,   1.66200,   1.00000]
#Rotated

# A = symbols("A")
# B = symbols("B")
# C = symbols("C")


# J=np.array([[A,C,0],
#             [C,A,0],
#             [0,0,B]])
# J_local = np.transpose(rotx(-90)) @ np.transpose(rotz(45)) @ J @rotz(45) @rotx(-90)

A=-0.75
B=-1.50
C=0.75

# A=1.0
# B=1.0
# C=0
J=np.array([[A,C,0],
            [C,A,0],
            [0,0,B]])
# J=np.array([[3,0,0],
#             [0,0,0],
#             [0,0,0]])
J_1=np.transpose(rotx(-90)) @ np.transpose(rotz(45)) @ J @rotz(45) @rotx(-90)
J_paper = np.array([[0,   0,    0],
                    [0,-1.5,    0],
                    [0,   0, -1.5]])
Jmatpa=np.round(J,3)

# Jmat=np.linalg.inv(roty(90)) @ J @roty(90)
# Jmat=np.linalg.inv(rotz(-45)) @ Jmat @rotz(-45)
# Jmatpp=np.round(Jmat,3)
Jmatpp=np.transpose(rotz(90))@J@rotz(90)
Jmatpp=np.round(Jmatpp,3)
# Jmatpp=np.round(Jmat,3)

g=np.array([[2.44, 0    , 0],
            [0   , 1.22 , 0],
            [0   , 0    , 2.14]])

# gmat=np.linalg.inv(roty(90)) @ g @roty(90)
gmat=np.transpose(rotz(45)) @ g@ rotz(45)
gmatpa=np.round(gmat,3)

# gmat=np.linalg.inv(roty(90)) @ g @roty(90)
gmat=np.transpose(rotz(-45)) @ g @rotz(-45)
gmatpp=np.round(gmat,3)


#[A,B,0],
#[B,C,E],
#[0,E,D]

# ev,ef,hamM=TT(Jmatpp)
expM001_1p8K=np.loadtxt("Ce_001_1p8K_MH.txt")
expM001_5K=np.loadtxt("MH001_5K.txt")
expM100_5K=np.loadtxt("MH100_5K.txt")
expM110_5K=np.loadtxt("MH110_5K.txt")

expM110_1p8K=np.loadtxt("Ce_110_1p8K_MH.txt")
expM100_1p8K=np.loadtxt("Ce_100_1p8K_MH.txt")

# print(np.round(TT(Jmatpa)[1], 3))

# tempH = 15
# H=np.array([tempH/np.sqrt(2), tempH/np.sqrt(2),0])
# ev,ef,hamM = TTM(Jmatpa,H,gmatpa)
# print(np.round(ev,3))
# print(np.round(ef))
#%%
def cal_mag(H,T):
        ev,ef,hamM=TTM(Jmatpa,H,gmatpa)
        # ev,ef,hamM=TTM(Jmatpp,H,gmatpp)
        # print(np.round(ef[:,1],3))
        egy=ev-ev[0]
        
        M1=0
        M2=0
        M=0
        
        #calculate partition function at temperature T
        # T=1.8
        Z=0
        kb=0.08617
        for n in range(0,len(egy)):
            Z+=np.exp(-egy[n]/(kb*T))
            
        for j in range(len(ef)):
            M1+=gmatpa @ (np.conj(ef[:,j]).T@S1@ef[:,j]) * np.exp(-egy[j]/(kb*T))/Z @H/np.linalg.norm(H)
            M2+=gmatpa @ (np.conj(ef[:,j]).T@S2@ef[:,j]) * np.exp(-egy[j]/(kb*T))/Z @H/np.linalg.norm(H)
    
        
        M1=np.linalg.norm(M1)
        M2=np.linalg.norm(M2)
        M=M1+M2
        return M1, M2, M,egy
    

def mag(**kwargs):
    energy1=[]
    energy2=[]
    energy3=[]
    field=[]
    Magnetization1=[]
    Magnetization2=[]
    Total_Magnetization=[]

    if kwargs['direction'] == "z":
        for i in np.arange(0.01,9.01,0.1):
            H=np.array([0,0,i])
            M1,M2,M,egy=cal_mag(H,kwargs['temp'])
            Magnetization1.append(M1)
            Magnetization2.append(M2)
            Total_Magnetization.append(M)
            energy1.append(egy[1])
            energy2.append(egy[2])
            energy3.append(egy[3])
            field.append(i)
    if kwargs['direction'] == "x":
        for i in np.arange(0.01,9.01,0.1):
            H=np.array([i,0,0])
            M1,M2,M,egy=cal_mag(H,kwargs['temp'])
            Magnetization1.append(M1)
            Magnetization2.append(M2)
            Total_Magnetization.append(M)
            energy1.append(egy[1])
            energy2.append(egy[2])
            energy3.append(egy[3])
            field.append(i)
    if kwargs['direction'] == "y":
        for i in np.arange(0.01,9.01,0.1):
            H=np.array([0,i,0])
            M1,M2,M,egy=cal_mag(H,kwargs['temp'])
            Magnetization1.append(M1)
            Magnetization2.append(M2)
            Total_Magnetization.append(M)
            energy1.append(egy[1])
            energy2.append(egy[2])
            energy3.append(egy[3])
            field.append(i)
    if kwargs['direction'] == "xy":
        for i in np.arange(0.01,9.01,0.1):
            H=np.array([i/np.sqrt(2),i/np.sqrt(2),0])
            M1,M2,M,egy=cal_mag(H,kwargs['temp'])
            Magnetization1.append(M1)
            Magnetization2.append(M2)
            Total_Magnetization.append(M)
            energy1.append(egy[1])
            energy2.append(egy[2])
            energy3.append(egy[3])
            field.append(i)
    if kwargs['direction'] == "x-y":
        for i in np.arange(0.01,9.01,0.1):
            H=np.array([i/np.sqrt(2),-i/np.sqrt(2),0])
            M1,M2,M,egy=cal_mag(H,kwargs['temp'])
            Magnetization1.append(M1)
            Magnetization2.append(M2)
            Total_Magnetization.append(M)
            energy1.append(egy[1])
            energy2.append(egy[2])
            energy3.append(egy[3])
            field.append(i)
    return Magnetization1, Magnetization2, Total_Magnetization,field,np.array([energy1,energy2,energy3])

#--------plotting the [0,0,1]-magnetization--------------
_,_,Total_Magnetization,field, energy= mag(direction="z",temp=1.8)
_,_,Total_Magnetization5,field, energy= mag(direction="z",temp=5)

# print(field)
plt.rcParams.update({
    "text.usetex": True
})
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

plt.plot(field,np.array(Total_Magnetization)*0.5,color="blue")
# plt.plot(field,np.array(Total_Magnetization5)*0.5,color="blue")
plt.plot(expM001_1p8K[:,0][::60]/10000,expM001_1p8K[:,1][::60],'.',color="blue",label="1.8K-001")
# plt.plot(expM001_5K[:,0][::60]/10000,10.9*expM001_5K[:,1][::60],'.',color="blue",label="5K-001")
plt.legend(loc="lower right")
# # plt.show()

#--------plotting the [1,0,0]-magnetization--------------
_,_,Total_Magnetization100,field,_= mag(direction="x",temp=1.8)
_,_,Total_Magnetization010,field,energy= mag(direction="y",temp=1.8)
mag100=np.array(Total_Magnetization100)+np.array(Total_Magnetization010)

_,_,Total_Magnetization100_5K,field,_= mag(direction="x",temp=5)
_,_,Total_Magnetization010_5K,field,energy= mag(direction="y",temp=5)
mag100_5K=np.array(Total_Magnetization100_5K)+np.array(Total_Magnetization010_5K)

plt.plot(field,mag100/4,color="green")
# plt.plot(field,mag100_5K/4,color="green")

plt.plot(expM100_1p8K[:,0][::60]/10000,expM100_1p8K[:,1][::60],'.',color="green",label="1.8K-100")
# plt.plot(expM100_5K[:,0][::60]/10000,10.9*expM100_5K[:,1][::60],'.',color="green",label="5K-100")
plt.legend(loc="lower right")
# plt.show()

# --------plotting the [1,1,0]-magnetization--------------
_,_,Total_Magnetization110,field,energy= mag(direction="xy",temp=1.8)
_,_,Total_Magnetization1_10,field,_= mag(direction="x-y",temp=1.8)
mag110=np.array(Total_Magnetization110)+np.array(Total_Magnetization1_10)

_,_,Total_Magnetization110_5K,field,energy= mag(direction="xy",temp=5)
_,_,Total_Magnetization1_10_5K,field,_= mag(direction="x-y",temp=5)
mag110_5K=np.array(Total_Magnetization110_5K)+np.array(Total_Magnetization1_10_5K)


# plt.plot(field,mag110/4,color="purple")
# plt.plot(field,mag110_5K/4,color="purple")
# plt.plot(expM110_1p8K[:,0][::60]/10000,expM110_1p8K[:,1][::60],'.',color="purple",label="1.8K-110")
# plt.plot(expM110_5K[:,0][::60]/10000,10.9*expM110_5K[:,1][::60],'.',color="purple",label="5K-110")
# SUH001=pd.read_csv("PRL\Sunny\H001.csv", delimiter=',')
# SUM001=pd.read_csv("PRL\Sunny\M001.csv", delimiter=',')
# SUH110=pd.read_csv("PRL\Sunny\H110.csv", delimiter=',')
# SUM110=pd.read_csv("PRL\Sunny\M110.csv", delimiter=',')
# SUM100=pd.read_csv("PRL\Sunny\M100.csv", delimiter=',')
# plt.plot(SUH001,SUM001/2,"-.",label='0K-001')
# plt.plot(SUH001,SUM100/2,"-.",label='0K-100',color='green')
# plt.plot(SUH110,SUM110/2/np.sqrt(2),"-.",label='0K-110',color='purple')


# plt.legend(loc="lower right")
# plt.xlabel("Magnetic Field (T)",fontsize=15)
# plt.ylabel(r'M($\mu_B$ per Ce)', fontsize=15)
# plt.show()

#%%--------Energy--------------

magfield1=[]
e0=[]
e1=[]
e2=[]
e3=[]
for i in np.arange(0,20,0.05):
    # i=9.7
    H=np.array([0,0,i])
    # H=np.array([i/np.sqrt(2),i/np.sqrt(2),0])
    ev,ef,hamM=TTM(Jmatpa,H,gmatpa)
    egy=ev-ev[0]
    if i > 8.5:
        ev = ev[[0,1,3,2]]
    # e0.append(egy[0])
    # e1.append(egy[1])
    # e2.append(egy[2])
    # e3.append(egy[3])
    e0.append(ev[0])
    e1.append(ev[1])
    e2.append(ev[2])
    e3.append(ev[3])
    
    magfield1.append(i)

magfield2=[]
e0_110=[]
e1_110=[]
e2_110=[]
e3_110=[]
for i in np.arange(0,20,0.05):
    # i=9.7
    # H=np.array([0,0,i])
    # H=np.array([0,i,0])
    H=np.array([i/np.sqrt(2),i/np.sqrt(2),0])
    ev,ef,hamM=TTM(Jmatpa,H,gmatpa)
    print(np.round(ef,3))
    egy=ev-ev[0]
    if i > 10.7:
        ev = ev[[1,0,3,2]]
    # e0.append(egy[0])
    # e1.append(egy[1])
    # e2.append(egy[2])
    # e3.append(egy[3])
    e0_110.append(ev[0])
    e1_110.append(ev[1])
    e2_110.append(ev[2])
    e3_110.append(ev[3])
    magfield2.append(i)

magfield3=[]
e0_1_10=[]
e1_1_10=[]
e2_1_10=[]
e3_1_10=[]
for i in np.arange(0,20,0.05):
    # i=9.7
    # H=np.array([0,0,i])
    # H=np.array([0,i,0])
    H=np.array([-i/np.sqrt(2),i/np.sqrt(2),0])
    ev,ef,hamM=TTM(Jmatpa,H,gmatpa)
    egy=ev-ev[0]
    if i > 7.5:
        ev = ev[[0,1,3,2]]
    # e0.append(egy[0])
    # e1.append(egy[1])
    # e2.append(egy[2])
    # e3.append(egy[3])
    e0_1_10.append(ev[0])
    e1_1_10.append(ev[1])
    e2_1_10.append(ev[2])
    e3_1_10.append(ev[3])
    magfield3.append(i)

#%%------------Checking Matrix Component------------------
matComponentsx=[]
matComponentsy=[]
matComponentsz=[]
f = []

for i in np.arange(9,13,1):
    H=np.array([i/np.sqrt(2),i/np.sqrt(2),0])
    ev,ef,hamM=TTM(Jmatpa,H,gmatpa)
    # ev,ef,hamM=TT(Jmatpa)
    egy=ev-ev[0]
    # print(np.real(np.round(ef,3)))
    matComponents = (np.conj(ef[:,0]).T@S1@ef[:,1])
    matComponentsx.append(matComponents[0])
    matComponentsy.append(matComponents[1])
    matComponentsz.append(matComponents[2])
    f.append(i)
f= np.array(f)
matComponentsx=np.array(matComponentsx)
matComponentsy=np.array(matComponentsy)
matComponentsz=np.array(matComponentsz)
plt.plot(f, matComponentsx.imag, label = 'x')
plt.plot(f, matComponentsy.imag,label = 'y')
plt.plot(f, matComponentsz,label='z')
plt.legend(loc = 'lower left')
plt.show()
#%%
i=11
H=np.array([i,0,0])
ev,ef,hamM=TTM(J_paper,H,g)
# ev,ef,hamM=TT(Jmatpa)
egy=ev-ev[0]
print(np.real(np.round(ef,3)))
matComponents = (np.conj(ef[:,0]).T@S1@ef[:,1])
print(np.round(matComponents,5))
#%%------------H||001------------------
fig, ax001 = plt.subplots(figsize = (4,8),dpi=600)
ax001.plot(magfield1,e0,'.', color = 'purple')
ax001.plot(magfield1,e1,'.', color = 'blue')
ax001.plot(magfield1,e2,'.',color='orange')
ax001.plot(magfield1,e3,'.',color= 'green')
# ax.set_ylim(0,2)
ax001.set_xticks([0,10,20],[0,10,20],fontsize=xlabelsize+5)
ax001.set_ylabel("$E$ (meV)", fontsize = ylabelsize+5)
for axis in ['top','bottom','left','right']:
    ax001.spines[axis].set_linewidth(3)
    ax001.tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5,labelsize=ylabelsize+5)
# ax0.annotate("(a)", xy=(0.02, 0.9),color = 'black', xycoords="axes fraction",fontsize=15)
ax001.annotate(r'$|\psi_3\rangle $', xy=(0.03, 0.75),color = 'green', xycoords="axes fraction",fontsize=30)
ax001.annotate(r'$|\psi_2\rangle $', xy=(0.03, 0.63),color = 'orange', xycoords="axes fraction",fontsize=30)
ax001.annotate(r'$|\psi_1\rangle $', xy=(0.03, 0.48),color = 'blue', xycoords="axes fraction",fontsize=30)
ax001.annotate(r'$|\psi_0\rangle $', xy=(0.03, 0.33),color = 'purple', xycoords="axes fraction",fontsize=30)
# ax001.annotate(r"$\mu_0\boldsymbol{\rm H}\parallel$ [0,0,1]", xy=(0.25, 0.92),color = 'black', xycoords="axes fraction",fontsize=20)
ax001.set_xlabel("$\mu_0H$ (T)", fontsize = xlabelsize+5)
plt.savefig(r'PRL\Fig2energy.svg', format='svg',bbox_inches = "tight")
#%%------------H||110------------------
fig, ax110 = plt.subplots(figsize = (6,4),dpi=600)
ax110.plot(magfield2,e0_110,'.', color = 'purple')
ax110.plot(magfield2,e1_110,'.', color = 'blue')
ax110.plot(magfield2,e2_110,'.',color='orange')
ax110.plot(magfield2,e3_110,'.',color= 'green')
# ax.set_ylim(0,2)
ax110.set_ylabel("$E$ (meV)", fontsize = ylabelsize+5)
ax110.set_xlabel("Magnetic Field (T)", fontsize = xlabelsize+5)
for axis in ['top','bottom','left','right']:
    ax110.spines[axis].set_linewidth(3)
    ax110.tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5,labelsize = 35)
# ax110.annotate("(b)", xy=(0.02, 0.9),color = 'black', xycoords="axes fraction",fontsize=15)
ax110.annotate(r"$|\psi_3\rangle $", xy=(0.05, 0.8),color = 'green', xycoords="axes fraction",fontsize=30)
ax110.annotate(r"$(\ \ \ \ \  \ \ \ \ \ \ \ \ \ )/\sqrt{2}$", xy=(0.215, 0.47),color = 'black', xycoords="axes fraction",fontsize=30)
ax110.annotate(r"$+$", xy=(0.397, 0.475),color = 'orange', xycoords="axes fraction",fontsize=30)
ax110.annotate(r"$-$", xy=(0.397, 0.43),color = 'blue', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_2\rangle $', xy=(0.475, 0.47),color = 'black', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_1\rangle $', xy=(0.255, 0.47),color = 'black', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_0\rangle $', xy=(0.05, 0.13),color = 'purple', xycoords="axes fraction",fontsize=30)
# ax110.annotate(r"Dimer A", xy=(0.35, 0.88),color = 'black', xycoords="axes fraction",fontsize=30)
plt.subplots_adjust(left=0.2, right=1.0, top=0.9, bottom=0.1)
plt.savefig(r'PRL\Fig3_110.svg', format='svg',bbox_inches = "tight")

#%%------------H||1_10------------------
fig, ax110 = plt.subplots(figsize = (6,4),dpi=600)
ax110.plot(magfield3,e0_1_10,'.', color = 'purple')
ax110.plot(magfield3,e1_1_10,'.', color = 'blue')
ax110.plot(magfield3,e2_1_10,'.',color='orange')
ax110.plot(magfield3,e3_1_10,'.',color= 'green')
# ax.set_ylim(0,2)
ax110.set_ylabel("$E$ (meV)", fontsize = ylabelsize+5)
ax110.set_xlabel("Magnetic Field (T)", fontsize = xlabelsize+5)
for axis in ['top','bottom','left','right']:
    ax110.spines[axis].set_linewidth(3)
    ax110.tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5, labelsize = 35)
# ax110.annotate("(b)", xy=(0.02, 0.9),color = 'black', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_3\rangle $', xy=(0.04, 0.75),color = 'green', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_2\rangle $', xy=(0.4, 0.8),color = 'orange', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_1\rangle $', xy=(0.26, 0.43),color = 'blue', xycoords="axes fraction",fontsize=30)
ax110.annotate(r'$|\psi_0\rangle $', xy=(0.04, 0.3),color = 'purple', xycoords="axes fraction",fontsize=30)
# ax110.annotate("Dimer B", xy=(0.35, 0.88),color = 'black', xycoords="axes fraction",fontsize=20)
plt.subplots_adjust(left=0.2, right=1.0, top=0.9, bottom=0.1)
plt.savefig(r'PRL\Fig3_1_10.svg', format='svg',bbox_inches = "tight")
#%%------------------Heat Capacity-------------------------------
# Cp0T=np.loadtxt("CpCe0T_001.txt")
Cp0T_110=np.loadtxt("Ce_Cp110_0T_normalized.txt")
Cp1p5T_110=np.loadtxt("Ce_Cp110_1p5T_normalized.txt")
Cp3T_110=np.loadtxt("Ce_Cp110_3T_normalized.txt")
Cp4p5T_110=np.loadtxt("Ce_Cp110_4p5T_normalized.txt")
Cp6T_110=np.loadtxt("Ce_Cp110_6T_normalized.txt")
Cp7p5T_110=np.loadtxt("Ce_Cp110_7p5T_normalized.txt")
Cp9T_110=np.loadtxt("Ce_Cp110_9T_normalized.txt")

Cp0T_001=np.loadtxt("Ce_Cp001_0T_normalized.txt")
Cp1p5T_001=np.loadtxt("Ce_Cp001_1p5T_normalized.txt")
Cp3T_001=np.loadtxt("Ce_Cp001_3T_normalized.txt")
Cp4p5T_001=np.loadtxt("Ce_Cp001_4p5T_normalized.txt")
Cp6T_001=np.loadtxt("Ce_Cp001_6T_normalized.txt")
Cp7p5T_001=np.loadtxt("Ce_Cp001_7p5T_normalized.txt")
Cp9T_001=np.loadtxt("Ce_Cp001_9T_normalized.txt")
def heat_capacity(H):
    # H=np.array([0,0,0])
    ev,ef,hamM=TTM(Jmatpa,H,gmatpa)
    egy=ev-ev[0]
    
    T=np.linspace(0.01,25.01,210)
    #calculate partition function at temperature T

    Z=0
    kb=0.08617
    for n in range(0,len(egy)):
        Z+=np.exp(-egy[n]/(kb*T))
    
    Cp=0
    
    T1=0
    for n in range(0,len(egy)):
        T1+=egy[n]**2 * np.exp(-egy[n]/(kb*T))
    T1=T1/Z
    
    T2=0
    for n in range(0,len(egy)):
        T2+=egy[n] * np.exp(-egy[n]/(kb*T))
    T2=T2/Z
    T2=T2**2
    
    Cp=(T1-T2)/(kb*T**2)
    Cp=8.134*11.60*Cp
    return T,Cp

E_CEF1=np.array([ 0.        ,  0.54453399, 52.50510963, 53.83482394, 60.30699842, 62.43177568])
E_CEF2=np.array([ 0.        ,  1.33146156, 53.42991578, 53.62062724, 61.71355619, 61.91240302])
def heat_capacity_CEF(E_CEF):
    egy=E_CEF
    
    T=np.linspace(0.01,25.01,210)
    #calculate partition function at temperature T

    Z=0
    kb=0.08617
    for n in range(0,len(egy)):
        Z+=np.exp(-egy[n]/(kb*T))
    
    Cp=0
    
    T1=0
    for n in range(0,len(egy)):
        T1+=egy[n]**2 * np.exp(-egy[n]/(kb*T))
    T1=T1/Z
    
    T2=0
    for n in range(0,len(egy)):
        T2+=egy[n] * np.exp(-egy[n]/(kb*T))
    T2=T2/Z
    T2=T2**2
    
    Cp=(T1-T2)/(kb*T**2)
    Cp=8.134*11.60*Cp
    return T,Cp


T,Cp0T001=heat_capacity(np.array([0,0,0]))
T,Cp1p5T001=heat_capacity(np.array([0,0,1.5]))
T,Cp3T001=heat_capacity(np.array([0,0,3]))
T,Cp4p5T001=heat_capacity(np.array([0,0,4.5]))
T,Cp6T001=heat_capacity(np.array([0,0,6]))
T,Cp7p5T001=heat_capacity(np.array([0,0,7.5]))
T,Cp9T001=heat_capacity(np.array([0,0,9]))


fig, axcp001 = plt.subplots(figsize = (10,4),dpi=600)
axcp001.plot(T,Cp0T001,color="blue")
# axcp110(T,(Cp1p5T110+Cp1p5T1_10)/2,color="red")
axcp001.plot(T,Cp3T001,color="green")
# axcp110.plot(T,(Cp4p5T110+Cp4p5T1_10)/2,color="purple")
axcp001.plot(T,Cp6T001,color="orange")
axcp001.plot(T, Cp7p5T001,color="black")
axcp001.plot(T,Cp9T001,color="purple")

alpha=0.90
axcp001.scatter(Cp0T_001[::3,0], alpha*Cp0T_001[::3,1],marker='o',facecolors='white', edgecolors="blue",zorder=2, label="0.0 T")
# axcp110.plot(Cp1p5T_110[:,0], alpha*Cp1p5T_110[:,1],'.',alpha=0.3,color="red")
axcp001.scatter(Cp3T_001[::3,0], alpha*Cp3T_001[::3,1],marker='s',facecolors='white',color="green",zorder=2,label="3.0 T")
# axcp110.scatter(Cp4p5T_110[:,0], alpha*Cp4p5T_110[:,1],marker='v',facecolors='white',color="purple",zorder=2,label="4.5T")
axcp001.scatter(Cp6T_001[::3,0], alpha*Cp6T_001[::3,1],marker='p',facecolors='white',color="orange",zorder=2,label="6.0 T")
axcp001.scatter(Cp7p5T_001[::3,0], alpha*Cp7p5T_001[::3,1],marker='P',facecolors='white',color="black",zorder=2, label="7.5 T")
axcp001.scatter(Cp9T_001[::3,0], alpha*Cp9T_001[::3,1],marker='*',facecolors='white',color="purple",zorder=2,label="9.0 T")
axcp001.legend(loc="upper right")
axcp001.set_xlim(0,15)
axcp001.set_ylim(-0.5,8.3)
axcp001.set_xlabel("Temperature (K)", fontsize=xlabelsize)
axcp001.set_xticks([0,5,10,15],['$0$', '$5$','$10$','$15$'])
axcp001.set_ylabel(r"$C_{\rm mag}$ (J/K/mol$_{\rm Ce}$)",fontsize=ylabelsize)
axcp001.legend(loc="center right",bbox_to_anchor=(0.85,0.15), frameon = False,ncol=3,columnspacing=0.0, fontsize=20)
axcp001.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5,labelsize = 25)
# axcp001.annotate(r"Dimer A + Dimer B", xy=(0.38, 0.92),color = 'black', xycoords="axes fraction",fontsize=20)
for axis in ['top','bottom','left','right']:
    axcp001.spines[axis].set_linewidth(3)
plt.savefig(r'PRL\SI-cp001.svg', format='svg',bbox_inches = "tight")

# plt.plot(T,Cp0T001,label="0T-001",color="blue")
# # plt.plot(T,Cp1p5T001,label="1.5T")
# plt.plot(T,Cp3T001,label="3T-001",color="pink")
# plt.plot(T,Cp4p5T001,label="4.5T-001",color="purple")
# plt.plot(T,Cp6T001,label="6T-001",color="orange")
# plt.plot(T,Cp7p5T001,label="7.5T-001",color="black")
# plt.plot(T,Cp9T001,label="9T-001",color="cyan")

# alpha=0.90
# plt.plot(Cp0T_001[:,0], alpha*Cp0T_001[:,1],'.',alpha=0.3,color="blue")
# # plt.plot(Cp1p5T_001[:,0], alpha*Cp1p5T_001[:,1],'.',alpha=0.3,color="red")
# plt.plot(Cp3T_001[:,0], alpha*Cp3T_001[:,1],'.',alpha=0.3,color="pink")
# plt.plot(Cp4p5T_001[:,0], alpha*Cp4p5T_001[:,1],'.',alpha=0.3,color="purple")
# plt.plot(Cp6T_001[:,0], alpha*Cp6T_001[:,1],'.',alpha=0.3,color="orange")
# plt.plot(Cp7p5T_001[:,0], alpha*Cp7p5T_001[:,1],'.',alpha=0.3,color="black")
# plt.plot(Cp9T_001[:,0], alpha*Cp9T_001[:,1],'.',alpha=0.3,color="cyan")

# plt.legend(loc="upper right")
# plt.xlim(0,25)
# plt.xlabel("Temperature (K)", fontsize=15)
# plt.ylabel(r"$C_{mag} (J/mol - FU \cdot K)$",fontsize=15)
# plt.show()

#%%-----------Fig3 Cp 110------------------------
T,Cp0T110=heat_capacity(np.array([0,0,0]))
T,Cp1p5T110=heat_capacity(np.array([1.5/np.sqrt(2),1.5/np.sqrt(2),0]))
T,Cp3T110=heat_capacity(np.array([3/np.sqrt(2),3/np.sqrt(2),0]))
T,Cp4p5T110=heat_capacity(np.array([4.5/np.sqrt(2),4.5/np.sqrt(2),0]))
T,Cp6T110=heat_capacity(np.array([6/np.sqrt(2),6/np.sqrt(2),0]))
T,Cp7p5T110=heat_capacity(np.array([7.5/np.sqrt(2),7.5/np.sqrt(2),0]))
T,Cp9T110=heat_capacity(np.array([9/np.sqrt(2),9/np.sqrt(2),0]))


T,Cp0T1_10=heat_capacity(np.array([0,0,0]))
T,Cp1p5T1_10=heat_capacity(np.array([1.5/np.sqrt(2), -1.5/np.sqrt(2), 0]))
T,Cp3T1_10=heat_capacity(np.array([3/np.sqrt(2), -3/np.sqrt(2), 0]))
T,Cp4p5T1_10=heat_capacity(np.array([4.5/np.sqrt(2), -4.5/np.sqrt(2), 0]))
T,Cp6T1_10=heat_capacity(np.array([6/np.sqrt(2), -6/np.sqrt(2), 0]))
T,Cp7p5T1_10=heat_capacity(np.array([7.5/np.sqrt(2), -7.5/np.sqrt(2), 0]))
T,Cp9T1_10=heat_capacity(np.array([9/np.sqrt(2), -9/np.sqrt(2), 0]))

fig, axcp110 = plt.subplots(figsize = (10,4),dpi=600)
axcp110.plot(T,(Cp0T110+Cp0T1_10)/2,color="blue")
# axcp110(T,(Cp1p5T110+Cp1p5T1_10)/2,color="red")
axcp110.plot(T,(Cp3T110+Cp3T1_10)/2,color="green")
# axcp110.plot(T,(Cp4p5T110+Cp4p5T1_10)/2,color="purple")
axcp110.plot(T,(Cp6T110+Cp6T1_10)/2,color="orange")
axcp110.plot(T, (Cp7p5T110+Cp7p5T1_10)/2,color="black")
axcp110.plot(T,(Cp9T110+Cp9T1_10)/2,color="purple")

alpha=0.90
axcp110.scatter(Cp0T_110[::3,0], alpha*Cp0T_110[::3,1],marker='o',facecolors='white', edgecolors="blue",zorder=2, label="0.0 T")
# axcp110.plot(Cp1p5T_110[:,0], alpha*Cp1p5T_110[:,1],'.',alpha=0.3,color="red")
axcp110.scatter(Cp3T_110[::3,0], alpha*Cp3T_110[::3,1],marker='s',facecolors='white',color="green",zorder=2,label="3.0 T")
# axcp110.scatter(Cp4p5T_110[:,0], alpha*Cp4p5T_110[:,1],marker='v',facecolors='white',color="purple",zorder=2,label="4.5T")
axcp110.scatter(Cp6T_110[::3,0], alpha*Cp6T_110[::3,1],marker='p',facecolors='white',color="orange",zorder=2,label="6.0 T")
axcp110.scatter(Cp7p5T_110[::3,0], alpha*Cp7p5T_110[::3,1],marker='P',facecolors='white',color="black",zorder=2, label="7.5 T")
axcp110.scatter(Cp9T_110[::3,0], alpha*Cp9T_110[::3,1],marker='*',facecolors='white',color="purple",zorder=2,label="9.0 T")
axcp110.legend(loc="upper right")
axcp110.set_xlim(0,15)
axcp110.set_ylim(-0.5,7.5)
axcp110.set_xlabel("Temperature (K)", fontsize=xlabelsize)
axcp110.set_xticks([0,5,10,15],['$0$', '$5$','$10$','$15$'])
axcp110.set_ylabel(r"$C_{\rm mag}$ (J/K/mol$_{\rm Ce}$)",fontsize=ylabelsize)
axcp110.legend(loc="center right",bbox_to_anchor=(0.85,0.15), frameon = False,ncol=3,columnspacing=0.0, fontsize=20)
axcp110.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5,labelsize = 25)
# axcp110.annotate(r"Dimer A + Dimer B", xy=(0.38, 0.92),color = 'black', xycoords="axes fraction",fontsize=20)
for axis in ['top','bottom','left','right']:
    axcp110.spines[axis].set_linewidth(3)
plt.savefig(r'PRL\Fig3cp.svg', format='svg',bbox_inches = "tight")
#%%--------------------------------CalCpDimerA-----------------------------------
fig, axcpA = plt.subplots(figsize = (6,4),dpi=600)
axcpA.plot(T,Cp0T110,color="blue",label="0 T")
# axcp110(T,(Cp1p5T110+Cp1p5T1_10)/2,color="red")
axcpA.plot(T,Cp3T110,color="green",label="3 T")
# axcp110.plot(T,(Cp4p5T110+Cp4p5T1_10)/2,color="purple")
axcpA.plot(T,Cp6T110,color="orange",label="6 T")
axcpA.plot(T, Cp7p5T110,color="black",label="7.5 T")
axcpA.plot(T,Cp9T110,color="purple", label="9.0 T")

axcpA.set_xlim(0,15)
axcpA.set_ylim(-0.5,8)

axcpA.set_xlabel("Temperature (K)", fontsize=xlabelsize+5)
axcpA.set_ylabel(r"$C_{\rm mag}$ (J/K/mol$_{\rm Ce}$)",fontsize=ylabelsize+5)
axcpA.legend(loc="upper right",bbox_to_anchor=(1.00,1.0), frameon = False,ncol=2,columnspacing=0.5, fontsize=20,handlelength=0.5)
axcpA.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5,labelsize = 35)
# axcp110.annotate(r"Dimer A + Dimer B", xy=(0.38, 0.92),color = 'black', xycoords="axes fraction",fontsize=20)
for axis in ['top','bottom','left','right']:
    axcpA.spines[axis].set_linewidth(3)
plt.savefig(r'PRL\Fig3cpA.svg', format='svg',bbox_inches = "tight")
#%%--------------------------------CalCpDimerB-----------------------------------
fig, axcpB = plt.subplots(figsize = (6,4),dpi=600)
axcpB.plot(T,Cp0T1_10,color="blue",label="0 T")
# axcp110(T,(Cp1p5T110+Cp1p5T1_10)/2,color="red")
axcpB.plot(T,Cp3T1_10,color="green",label="3 T")
# axcp110.plot(T,(Cp4p5T110+Cp4p5T1_10)/2,color="purple")
axcpB.plot(T,Cp6T1_10,color="orange",label="6 T")
axcpB.plot(T, Cp7p5T1_10,color="black",label="7.5 T")
axcpB.plot(T,Cp9T1_10,color="purple", label="9.0 T")

axcpB.set_xlim(0,15)
axcpB.set_ylim(-0.5,8)
axcpB.set_xlabel("Temperature (K)", fontsize=xlabelsize+5)
axcpB.set_ylabel(r"$C_{\rm mag}$ (J/K/mol$_{\rm Ce}$)",fontsize=ylabelsize+5)
# axcpB.legend(loc="center right",bbox_to_anchor=(0.8,0.15), frameon = False,ncol=2,columnspacing=0.0, fontsize=15)
axcpB.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5,labelsize = 35)
# axcp110.annotate(r"Dimer A + Dimer B", xy=(0.38, 0.92),color = 'black', xycoords="axes fraction",fontsize=20)
for axis in ['top','bottom','left','right']:
    axcpB.spines[axis].set_linewidth(3)
plt.savefig(r'PRL\Fig3cpB.svg', format='svg',bbox_inches = "tight")
#%%###################################################################
#   We calculate single crystal diffraction patter within HK0 plane.
#   v is the excited levels. v=0,1,2,3
######################################################################

def SQ(h,k,l,v,atom1,atom2,Jmat,H,g):
    
    """"SQ for single crystal """
    T=TTM(Jmat,H,g)
    muBT=5.7883818012e-2
    ef=T[1]
    S=np.array([rotS(g,S1),rotS(g,S2)])
    # S=np.array([S1,S2])
    R=np.array([atom1,atom2])
    Omega=0
    Q=np.array([h,k,l]) @ np.array([b1,b2,b3])
    Qnorm=np.linalg.norm(Q)

    for a in range(0,3):
        for b in range(0,3):
            for m in range(0,2):
                for n in range(0,2):
                    Omega=Omega+((kron_delta(a,b)-Q[a]*Q[b]/(Qnorm+0.0000000001)**2) *  np.exp(1j*2*pi*np.dot( [h,k,l],(R[m]-R[n])) ) *\
                                np.conj(ef[:,0]).T @ S[m][a] @ ef[:,v] * np.conj(ef[:,v]).T @S[n][b] @ ef[:,0] )
    Omega=Omega*FF(Qnorm)**2
    return Omega

def hkl_SQ(atom1,atom2,Jmat,H,gmat,v):
    '''Calculating the S(q,w) between two specific transitions
       Currently set up to calculate the HK0 plane wiht L set to 0.
       
       atom1, atom2 = positional vectors of the two dimer atoms
       
       Jmat = Exchange matrices.
       
       H = magnetic field
       
       gmat = g-tensor matrix
       
       v = transition level from ground state (0 level).v can be 1, 2, 3, 4
    '''
    
    h=np.linspace(-2.7,2.7,81)
    k=np.linspace(-2.7,2.7,81)
    I_mat=np.zeros([len(h),len(k)])
    for i in range(len(h)):
        for j in range(len(k)):
            I_mat[i,j]=SQ(h[i],k[j],0,v,atom1,atom2, Jmat, H, gmat)
    # plt.imshow(I_mat,origin='lower',cmap='jet')
    return I_mat
    # plt.savefig(r'C:\Users\qmc\OneDrive\ONRL\Data\level3.png', format='png',dpi=400)

# I=hkl_SQ(R2,R3,Jmatpa,np.array([0,0,0]),gmatpa,1)+hkl_SQ(R1,R4,Jmatpp,np.array([0,0,0]),gmatpa,1)
# plt.imshow(I,origin='lower',cmap='jet',vmin=0.0,vmax=1.0)

def HH0(atom1, atom2, Jmat, H, gmat, v):
    h=np.linspace(0.2,1,61)
    I_mat=np.zeros([len(h)])
    for i in range(len(h)):
        I_mat[i]=SQ(h[i],h[i],0,v,atom1,atom2, Jmat, H, gmat)
    # plt.imshow(I_mat,origin='lower',cmap='jet')
    return h, I_mat

def int_HHL(atom1, atom2, Jmat, H, gmat, v, pm, l):
    '''pmh is the sign determine HH or H-H, l is the intercept shifting HH up and down'''
    h=np.linspace(0.0,2.5,41)
    I_mat=np.zeros([len(h)])
    for i in range(len(h)):
        I_mat[i]=SQ(h[i]-pm*l,pm*h[i]+l,0,v,atom1,atom2, Jmat, H, gmat)
    # plt.imshow(I_mat,origin='lower',cmap='jet')
    return h, I_mat

def integral(atom1, atom2, Jmat, H, gmat, v, pm, s,t,step):
    intensity=0
    for i in np.arange(s,t,step):
        h, I_mat=int_HHL(atom1,atom2,Jmat,H,gmat,v, pm, i)
        intensity+=I_mat
    return h, intensity

def fieldplot(B):
    H=np.array([0,0,B])
    ev,ef,hamM=TTM(Jmatpp,H,gmatpp)
    egy=ev-ev[0]
    ef=(np.round(ef,3))
    print(egy.round(3))

    #Plotting 4 panels of transverse and longitudinal component
    I_3=hkl_SQ(R2,R3,Jmatpa,H,gmatpa,3)+hkl_SQ(R1,R4,Jmatpp,H,gmatpp,3)
    I_2=hkl_SQ(R2,R3,Jmatpa,H,gmatpa,2)+hkl_SQ(R1,R4,Jmatpp,H,gmatpp,2)
    I_1=hkl_SQ(R2,R3,Jmatpa,H,gmatpa,1)+hkl_SQ(R1,R4,Jmatpp,H,gmatpp,1)

    tick_locs=[8 , 24, 40, 56, 72]
    tick_lbls=[-2, -1, 0 , 1 , 2]
    # tick_locs=[0 , 20, 40, 60, 80]
    # tick_lbls=[-2.5, -1.5, 0 , 2 , 2.5]
    fig,ax=plt.subplots(2,2,figsize=(19,16), dpi = 200)
    mi=0
    if B==0:
        ma=8
    else:
        ma=5#use 16 for 0T, 10 for 4T
    a1=ax[0,0].imshow(I_3,origin='lower',cmap='jet',vmin=mi,vmax=ma/1)
    ax[0,0].set_title(f'$\psi_3$:$E$={np.round(egy[3].real,2)} meV',fontsize=30,y=1.02)
    cbar1=fig.colorbar(a1,ax=ax[0,0],fraction=0.046, pad=0.04)
    cbar1.set_label("Intensity (arb. unit)",rotation = 270, fontsize = 30,labelpad = 35)
    cbar1.ax.tick_params(labelsize=20)
    
    a2=ax[0,1].imshow(I_2,origin='lower',cmap='jet',vmin=mi,vmax=1*ma)
    ax[0,1].set_title(f'$\psi_2$:$E$={np.round(egy[2].real,2)} meV', fontsize=30,y=1.02)
    cbar2=fig.colorbar(a2,ax=ax[0,1],fraction=0.046, pad=0.04)
    cbar2.set_label("Intensity (arb. unit)",rotation = 270, fontsize = 30,labelpad = 35)
    cbar2.ax.tick_params(labelsize=20)
    
    a3=ax[1,1].imshow(I_1,origin='lower',cmap='jet',vmin=mi,vmax=2*ma)
    ax[1,1].set_title(f'$\psi_1$:$E$={np.round(egy[1].real,2)} meV',fontsize=30,y=1.02)
    cbar3=fig.colorbar(a3,ax=ax[1,1],fraction=0.046, pad=0.04)
    cbar3.set_label("Intensity (arb. unit)",rotation = 270, fontsize = 30,labelpad = 35)
    cbar3.ax.tick_params(labelsize=20)
    
    if math.isclose(egy[2], egy[3], rel_tol=0.3):
        a4=ax[1,0].imshow(I_2+I_3,origin='lower',cmap='jet',vmin=mi,vmax=ma) ########
    elif math.isclose(egy[2], egy[1], rel_tol=0.3):
        a4=ax[1,0].imshow(I_2+I_1,origin='lower',cmap='jet',vmin=mi,vmax=2*ma)
    ax[1,0].set_title('$\psi_1$ + $\psi_2$ ',fontsize=30,y=1.02)
    cbar4=fig.colorbar(a4,ax=ax[1,0],fraction=0.046, pad=0.04)
    cbar4.set_label("Intensity (arb. unit)",rotation = 270, fontsize = 30,labelpad = 35)
    cbar4.ax.tick_params(labelsize=20)

    for i in range(0,2):
        for j in range(0,2):
            for axis in ['top','bottom','left','right']:
                ax[i,j].spines[axis].set_linewidth(3)
            ax[i,j].tick_params(width=4,direction='in',length=8,right=True,top=True,labelsize=15,pad=10)
            ax[i,j].set_xticks(tick_locs,tick_lbls, fontsize=30)
            ax[i,j].set_yticks(tick_locs,tick_lbls, fontsize=30)
            if j == 0:
                ax[i,j].set_ylabel('[0,K,0]', fontsize = 35)
            if i == 1:
                ax[i,j].set_xlabel('[H,0,0]', fontsize = 35)
    plt.show()
    return I_3, I_2, I_1, ef, egy
    

hh=1
# ef0=fieldplot(0)
# ef0=fieldplot(1)
# ef0=fieldplot(2)
# ef0=fieldplot(3)
I_30, I_20, I_10, ef0, egy0=fieldplot(0)
I_34, I_24, I_14, ef4, egy4=fieldplot(4)

#%%

#37 - 44 in h is -0.1875 - 0.1875 

def gaussian(x,Area, width, pos):
    pi = np.pi
    Y = Area/(width*np.sqrt(2*pi)) * np.exp(-(x-pos)**2/2/width**2)
    return Y

def EvsH(B):
    I_3, I_2, I_1, ef, egy = fieldplot(B)
    pwd3 = []
    pwd2 = []
    pwd1 = []
    for i in range(len(I_3)):
        intensity3 = 0
        intensity2 = 0
        intensity1 = 0
        for j in range(37,45):
            intensity3 += I_3[j][i]
            intensity2 += I_2[j][i]
            intensity1 += I_1[j][i]
        pwd3.append(intensity3)
        pwd2.append(intensity2)
        pwd1.append(intensity1)
            
    E=np.arange(0,2.0,0.01)#energy
    yspace3=[]
    for i in range(len(pwd3)):
        yspace3.append(gaussian(E,pwd3[i],0.01, egy[3]))
    yspace3 = np.array(yspace3)
    
    yspace2=[]
    for i in range(len(pwd2)):
        yspace2.append(gaussian(E,pwd2[i],0.01, egy[2]))
    yspace2 = np.array(yspace2)
    
    yspace1=[]
    for i in range(len(pwd1)):
        yspace1.append(gaussian(E,pwd1[i],0.01, egy[1]))
    yspace1 = np.array(yspace1)
    
    yspace_all = np.transpose(yspace1+yspace2+yspace3)
    # plt.imshow(yspace_all, origin = 'lower', vmin =0, vmax = 1000, cmap = 'jet')
    yspace_resize = cv2.resize(yspace_all, (92*2, 67))
    yspace_resize = cv2.GaussianBlur(yspace_resize, (5, 5),0)
    # plt.imshow(yspace_resize[:,92::], origin = 'lower', vmin =0, vmax = 1300, cmap = 'jet')
    return yspace_resize

#-----------0 T data------------------------------
yspace_resizeE0T = EvsH(0)
plt.rcParams['figure.dpi'] = 500

dataE0T=np.loadtxt("0TK0.2.txt")
for i in range(len(dataE0T)):
    for j in range(len(dataE0T[0])):
        if dataE0T[i][j] < -1:
            dataE0T[i][j] = np.nan
alldata0T=np.hstack((dataE0T[0:len(dataE0T)//2,:], yspace_resizeE0T[:,92::]/1.5e5))
alldata0T= cv2.resize(alldata0T, (150,81))
# plt.imshow(alldata0T, origin = 'lower', vmin =0, vmax =0.01, cmap = 'jet')
# plt.show()

dataE0THK1=np.loadtxt("0THK0.4_1.txt")
for i in range(len(dataE0THK1)):
    for j in range(len(dataE0THK1[0])):
        if dataE0THK1[i][j] < -1:
            dataE0THK1[i][j] = np.nan
dataE0THK1 = dataE0THK1[0:len(dataE0THK1)//2,:][5:134, 5:134] # resize to be between -2.5 to 2.5 in HK
dataE0THK1 = cv2.resize(dataE0THK1, (81,81))
alldata0THK1= np.vstack((dataE0THK1,(I_20+I_10)/1400))
# plt.imshow(alldata0THK1,vmin= 0 , vmax = 0.012, cmap = 'jet')
# plt.show()

dataE0THK2=np.loadtxt("0THK0.4_2.txt")
for i in range(len(dataE0THK2)):
    for j in range(len(dataE0THK2[0])):
        if dataE0THK2[i][j] < -1:
            dataE0THK2[i][j] = np.nan
dataE0THK2 = dataE0THK2[0:len(dataE0THK2)//2,:]
dataE0THK2 = cv2.resize(dataE0THK2, (81,81))
alldata0THK2= np.vstack((dataE0THK2/0.1,I_30/80))
# plt.imshow(alldata0THK2,vmin= 0 , vmax = 0.05, cmap = 'jet')
# plt.show()

#-----------5 T data------------------------------
yspace_resizeE4T = EvsH(4.2)

dataE4T=np.loadtxt("4TK0.2.txt")
for i in range(len(dataE4T)):
    for j in range(len(dataE4T[0])):
        if dataE4T[i][j] < -1:
            dataE4T[i][j] = np.nan
dataE4T = cv2.resize(dataE4T, (92,134))
alldata4T=np.hstack((dataE4T[0:len(dataE4T)//2,:], yspace_resizeE4T[:,92::]/8e4))
alldata4T= cv2.resize(alldata4T, (150,81))
# plt.imshow(alldata4T, origin = 'lower', vmin =0, vmax =0.01, cmap = 'jet')
# plt.show()

dataE4THK1=np.loadtxt("4THK0.4_1.txt")
for i in range(len(dataE4THK1)):
    for j in range(len(dataE4THK1[0])):
        if dataE4THK1[i][j] < -1:
            dataE4THK1[i][j] = np.nan
dataE4THK1 = dataE4THK1[0:len(dataE4THK1)//2,:][4:132, 4:132] # resize to be between -2.5 to 2.5 in HK
dataE4THK1 = cv2.resize(dataE4THK1, (81,81))
alldata4THK1= np.vstack((dataE4THK1,(I_14)/1000))
# plt.imshow(alldata4THK1,vmin= 0 , vmax = 0.010, cmap = 'jet')
# plt.show()

dataE4THK2=np.loadtxt("4THK0.4_2.txt")
for i in range(len(dataE4THK2)):
    for j in range(len(dataE4THK2[0])):
        if dataE4THK2[i][j] < -1:
            dataE4THK2[i][j] = np.nan
dataE4THK2 = dataE4THK2[0:len(dataE4THK2)//2,:][2:129, 1:129] # resize to be between -2.5 to 2.5 in HK
dataE4THK2 = cv2.resize(dataE4THK2, (81,81))
alldata4THK2= np.vstack((dataE4THK2,(I_24)/1000))
# plt.imshow(alldata4THK2,vmin= 0 , vmax = 0.0060, cmap = 'jet')
# plt.show()

dataE4THK3=np.loadtxt("4THK0.4_3.txt")
for i in range(len(dataE4THK3)):
    for j in range(len(dataE4THK3[0])):
        if dataE4THK3[i][j] < -1:
            dataE4THK3[i][j] = np.nan
dataE4THK3 = dataE4THK3[0:len(dataE4THK3)//2,:] # resize to be between -2.5 to 2.5 in HK
dataE4THK3 = cv2.resize(dataE4THK3, (81,81))
alldata4THK3= np.vstack((dataE4THK3,(I_34)/1000))
# plt.imshow(alldata4THK3,vmin= 0 , vmax = 0.0060, cmap = 'jet')
# plt.show()

#%%---------------------- Neutron data-------------------
inner1 = [['innerA', 'innerB','innerC']]
inner2 = [['innerD', 'innerE','innerF']]
outer = [['upper left',  inner1],
          ['lower left', inner2]]
fig, ax = plt.subplot_mosaic(outer,figsize = (21.55,10.6), layout = 'constrained',dpi = 1000, gridspec_kw={'wspace': 0.0}, width_ratios=[1.0,0.90])
pos1 = ax['upper left'].imshow(alldata0T, origin = 'lower', vmin =0, vmax =0.01, cmap = 'jet')
ax['upper left'].annotate('(a)', xy=(0.01, 0.90), xycoords='axes fraction',fontsize = 30)
ax['upper left'].annotate('H = 0 T', xy=(0.17, 0.90), xycoords='axes fraction',fontsize = 30, color = 'white')

ax['innerA'].imshow(alldata0THK2,vmin= 0 , vmax = 0.05, cmap = 'jet')
ax['innerA'].annotate('(b)', xy=(0.02, 0.95), xycoords='axes fraction',fontsize = 17)

ax['innerB'].imshow(alldata0THK1,vmin= 0 , vmax = 0.012, cmap = 'jet')
ax['innerB'].annotate('(c)', xy=(0.02, 0.95), xycoords='axes fraction',fontsize = 17)

pos3 = ax['innerC'].imshow(np.vstack((I_10, I_20)),vmin= 0 , vmax = 15, cmap = 'jet')
ax['innerC'].annotate('(d)', xy=(0.02, 0.95), xycoords='axes fraction',fontsize = 17, color = 'white')

pos2 = ax['lower left'].imshow(alldata4T, origin = 'lower', vmin =0, vmax =0.01, cmap = 'jet')
ax['lower left'].annotate('(e)', xy=(0.01, 0.90), xycoords='axes fraction',fontsize = 30)
ax['lower left'].annotate(r'$H$ = 4 T', xy=(0.17, 0.90), xycoords='axes fraction',fontsize = 30, color = 'white')


pos4 = ax['innerF'].imshow(alldata4THK1,vmin= 0 , vmax = 0.010, cmap = 'jet')
ax['innerF'].annotate('(h)', xy=(0.02, 0.95), xycoords='axes fraction',fontsize = 17)


ax['innerE'].imshow(alldata4THK2,vmin= 0 , vmax = 0.0060, cmap = 'jet')
ax['innerE'].annotate('(g)', xy=(0.02, 0.95), xycoords='axes fraction',fontsize = 17)

pos4 = ax['innerD'].imshow(alldata4THK3,vmin= 0 , vmax = 0.0060, cmap = 'jet')
ax['innerD'].annotate('(f)', xy=(0.02, 0.95), xycoords='axes fraction',fontsize = 17)


for i in ["upper left", "lower left"]:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,labelsize=15,pad=5)
    tick_locs=[10  ,  25 ,   41 ,  58 ,  74 , 90 ,  108, 126, 144, 162, 180]
    tick_lbls=[-2.5, -2.0,  -1.5, -1.0, -0.5,  0 ,  0.5, 1.0, 1.5, 2.0, 2.5]
    E_loc = [0,  20,  40,  60, 80]
    E_lbls= [0, 0.5, 1.0, 1.5, 2.0]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=25)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=25)
    ax[i].set_ylabel("Energy (meV)", fontsize = 30)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['lower left'].set_xlabel('[H,0,0]', fontsize = 30)

for i in ['innerA', 'innerB','innerC','innerD', 'innerE','innerF']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,labelsize=15,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=[-2, 1,    0, 1, 2 ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= [2, 1,  0, -1, -2,  2,   1,  0,   -1, -2 ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=20)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=20)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerA'].set_ylabel("[0,K,0]", fontsize = 25)
ax['innerD'].set_ylabel("[0,K,0]", fontsize = 25)
for i in ['innerD', 'innerE','innerF']:
    ax[i].set_xlabel('[H,0,0]', fontsize = 25)

cb2=fig.colorbar(pos3, ax=ax['innerC'],location = 'right',\
                  shrink = 0.85, pad =0.15, aspect = 30,\
                  ticks = [0,3,6,9,12,15])
cb2.ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize = 20)
cb2.set_label('Intensity (arb.unit)', fontsize = 25)

cb3=fig.colorbar(pos4, ax=ax['innerF'],location = 'right',\
                  shrink = 0.85, pad =0.15, aspect = 30,\
                    ticks = [0,0.0012,0.0024,0.0036,0.0048,0.006])
cb3.ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1], fontsize = 20)
cb3.set_label('Intensity (arb.unit)', fontsize = 25)
plt.savefig(r'PRL\neutron.svg', format='svg')
#%%-------------Neutron Data--------------------
fig, ax['upper left'] = plt.subplots(figsize = (8,8),dpi = 600,constrained_layout=True)
pos1 = ax['upper left'].imshow(alldata0T, origin = 'lower', vmin =0, vmax =0.01, cmap = 'jet')
ax['upper left'].annotate(r'$\mu_0H$ = 0 T', xy=(0.78,0.05), xycoords='axes fraction',fontsize = 20, color = 'white')
# cb1=fig.colorbar(pos1, ax=ax['upper left'],location = 'top',\
#                   shrink = 0.5, pad =0.1, aspect = 35,\
#                   ticks = [0,0.002,0.004,0.006,0.008,0.01])
# cb1.ax.set_xticklabels([0,0.2,0.4,0.6,0.8,1], fontsize = 30)
# cb1.set_label('Intensity (arb.unit)', fontsize = 20)
for i in ["upper left"]:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,labelsize=15,pad=5)
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,labelsize=15,pad=5)
    r=1.5
    tick_locs=[(0.2/5.4)*150, (1.2/5.4)*150, (2.2/5.4)*150, (3.2/5.4)*150, (4.2/5.4)*150, (5.2/5.4)*150] # changed calculation to 2.7remember chagne it back for other plots
    tick_lbls=['$-2.5$',  '$-1.5$', '$-0.5$', '$0.5$', '$1.5$', '$2.5$']
    E_loc = [0,  20,  40,  60, 80]
    E_lbls= [0, 0.5, 1.0, 1.5, 2.0]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=25)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=25)
    ax[i].set_ylabel("Energy transfer (meV)", fontsize = 25)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
    ax[i].set_xlabel('[$h$,0,0]', fontsize = 25)
plt.savefig(r'PRL/neutron1.svg', format='svg', bbox_inches = "tight")

fig, ax['lower left'] = plt.subplots(figsize = (8,8),dpi = 600,constrained_layout=True)
pos2 = ax['lower left'].imshow(alldata4T, origin = 'lower', vmin =0, vmax =0.01, cmap = 'jet')
ax['lower left'].annotate(r'$\mu_0H$ = 4 T', xy=(0.78,0.05), xycoords='axes fraction',fontsize = 20, color = 'white')
for i in ["lower left"]:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,labelsize=15,pad=5)
    r=1.5
    tick_locs=[6*r, 24*r, 42*r, 60*r, 80*r, 100*r]
    tick_lbls=[-2.5,  -1.5, -0.5,   0.5, 1.5, 2.5]
    E_loc = [0,  20,  40,  60, 80]
    E_lbls= [0, 0.5, 1.0, 1.5, 2.0]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=25)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=25)
    ax[i].set_ylabel("Energy transfer (meV)", fontsize = 25)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
    ax[i].set_xlabel('[$h$,0,0]', fontsize = 25)
plt.savefig(r'PRL/neutron2.svg', format='svg', bbox_inches = "tight")
#%%---------dynamic structure factors---------------
dynamicfontsize=30
fig, ax['innerA'] = plt.subplots(figsize = (6,6),dpi = 600,constrained_layout=True)
ax['innerA'].imshow(alldata0THK2,vmin= 0 , vmax = 0.05, cmap = 'jet')
for i in ['innerA']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=['$-2$', '$-1$',    '$0$', '$1$', '$2$' ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= ['$2$', '$1$',  '$0$', '$-1$', '$-2$',  '$2$',   '$1$',  '$0$',   '$-1$', '$-2$' ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=dynamicfontsize)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=dynamicfontsize)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerA'].set_ylabel("[0,$k$,0]", fontsize = ylabelsize+5)
ax['innerA'].set_xlabel('[$h$,0,0]', fontsize = xlabelsize+5)
plt.savefig(r'PRL/neutronA.svg', format='svg', bbox_inches = "tight")

fig, ax['innerB'] = plt.subplots(figsize = (6,6),dpi = 600,constrained_layout=True)
ax['innerB'].imshow(alldata0THK1,vmin= 0 , vmax = 0.012, cmap = 'jet')
for i in ['innerB']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=['$-2$', '$-1$',    '$0$', '$1$', '$2$' ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= ['$2$', '$1$',  '$0$', '$-1$', '$-2$',  '$2$',   '$1$',  '$0$',   '$-1$', '$-2$' ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=dynamicfontsize)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=dynamicfontsize)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerB'].set_ylabel("[0,$k$,0]", fontsize = ylabelsize+5)
ax['innerB'].set_xlabel('[$h$,0,0]', fontsize = xlabelsize+5)
plt.savefig(r'PRL/neutronB.svg', format='svg', bbox_inches = "tight")

fig, ax['innerC'] = plt.subplots(figsize = (6,6),dpi = 600,constrained_layout=True)
ax['innerC'].imshow(np.vstack((I_10, I_20)),vmin= 0 , vmax = 15, cmap = 'jet')
for i in ['innerC']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=['$-2$', '$-1$',    '$0$', '$1$', '$2$' ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= ['$2$', '$1$',  '$0$', '$-1$', '$-2$',  '$2$',   '$1$',  '$0$',   '$-1$', '$-2$' ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=dynamicfontsize)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=dynamicfontsize)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerC'].set_ylabel("[0,$k$,0]", fontsize = ylabelsize+5)
ax['innerC'].set_xlabel('[$h$,0,0]', fontsize = xlabelsize+5)
plt.savefig(r'PRL/neutronC.svg', format='svg', bbox_inches = "tight")

fig, ax['innerF'] = plt.subplots(figsize = (6,6),dpi = 600,constrained_layout=True)
ax['innerF'].imshow(alldata4THK1,vmin= 0 , vmax = 0.010, cmap = 'jet')
for i in ['innerF']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=['$-2$', '$-1$',    '$0$', '$1$', '$2$' ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= ['$2$', '$1$',  '$0$', '$-1$', '$-2$',  '$2$',   '$1$',  '$0$',   '$-1$', '$-2$' ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=dynamicfontsize)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=dynamicfontsize)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerF'].set_ylabel("[0,$k$,0]", fontsize = ylabelsize+5)
ax['innerF'].set_xlabel('[$h$,0,0]', fontsize = xlabelsize+5)
plt.savefig(r'PRL/neutronF.svg', format='svg', bbox_inches = "tight")

fig, ax['innerE'] = plt.subplots(figsize = (6,6),dpi = 600,constrained_layout=True)
ax['innerE'].imshow(alldata4THK2,vmin= 0 , vmax = 0.0060, cmap = 'jet')
for i in ['innerE']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=['$-2$', '$-1$',    '$0$', '$1$', '$2$' ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= ['$2$', '$1$',  '$0$', '$-1$', '$-2$',  '$2$',   '$1$',  '$0$',   '$-1$', '$-2$' ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=dynamicfontsize)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=dynamicfontsize)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerE'].set_ylabel("[0,$k$,0]", fontsize = ylabelsize+5)
ax['innerE'].set_xlabel('[$h$,0,0]', fontsize = xlabelsize+5)
plt.savefig(r'PRL/neutronE.svg', format='svg', bbox_inches = "tight")

fig, ax['innerD'] = plt.subplots(figsize = (6,6),dpi = 600,constrained_layout=True)
ax['innerD'].imshow(alldata4THK3,vmin= 0 , vmax = 0.0060, cmap = 'jet')
for i in ['innerD']:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,pad=5)
    tick_locs=[8,  24,  40, 56,  72]
    tick_lbls=['$-2$', '$-1$',    '$0$', '$1$', '$2$' ]
    E_loc = [8,24, 40, 56, 72, 89, 105, 120, 135, 151]
    E_lbls= ['$2$', '$1$',  '$0$', '$-1$', '$-2$',  '$2$',   '$1$',  '$0$',   '$-1$', '$-2$' ]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=dynamicfontsize)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=dynamicfontsize)
    # ax[i].set_xlabel('[H,0,0]', fontsize = 15)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
ax['innerD'].set_ylabel("[0,$k$,0]", fontsize = ylabelsize+5)
ax['innerD'].set_xlabel('[$h$,0,0]', fontsize = xlabelsize+5)
plt.savefig(r'PRL/neutronD.svg', format='svg', bbox_inches = "tight")

#%%-------------Fig1-------------------
fig, ax = plt.subplots(2,3, figsize = (21.55,10.8), layout = 'constrained',dpi = 400)

scale=1.1
B001 = np.load('PRL/CEF/CEFB001.npy')
M001 = np.load('PRL/CEF/CEFM001.npy')
B100 = np.load('PRL/CEF/CEFB100.npy')
M100 = np.load('PRL/CEF/CEFM100.npy')
M010 = np.load('PRL/CEF/CEFM010.npy')
B110 = np.load('PRL/CEF/CEFB110.npy')
M110x = np.load('PRL/CEF/CEFM110x.npy')
M110y = np.load('PRL/CEF/CEFM110y.npy')

# scale=1.0
# ax[1,0].plot(B001, scale*M001,'--',color='blue')

# ax[1,0].plot(B100, scale*(M100+M010)/2,"--",color='purple',linewidth=5)

# ax[1,0].plot(np.sqrt(2)*B110, scale*(M110x+M110y)/np.sqrt(2),'--',color='green',linewidth=5)

ax[1,0].plot(field,np.array(Total_Magnetization)*0.5,color="blue",zorder=0)
ax[1,0].plot(expM001_1p8K[:,0][::60]/10000,expM001_1p8K[:,1][::60],'o',color="blue",label="$H$ || [0,0,1]",zorder=1)
ax[1,0].legend(loc="lower right", frameon = False)
ax[1,0].plot(field,mag100/4,color="green", zorder=0)
ax[1,0].plot(expM100_1p8K[:,0][::60]/10000,expM100_1p8K[:,1][::60],'v',color="green",label="$H$ || [1,0,0]",zorder=1)
ax[1,0].legend(loc="lower right", frameon = False)
ax[1,0].plot(field,mag110/4,color="purple", zorder=0)
ax[1,0].plot(expM110_1p8K[:,0][::60]/10000,expM110_1p8K[:,1][::60],'s',color="purple",label="$H$ || [1,1,0]",zorder=1)
ax[1,0].legend(loc="lower right", frameon = False)
ax[1,0].set_xlabel("Magnetic Field (T)",fontsize=xlabelsize)
ax[1,0].set_ylabel(r'M ($\mu_B$/Ce)', fontsize=ylabelsize)
ax[1,0].annotate('$T$ = 1.8 K', xy=(0.65, 0.35), xycoords='axes fraction',fontsize = annotatesize)
# ax[1,0].annotate('(d)', xy=(0.02, 0.9), xycoords='axes fraction',fontsize = plotlabelsize)

aR = np.array([ax.ravel()[4].twinx()])
ax[1,1].plot(T,Cp0T001/T/2,color="blue")
S=[]
ST=[]
entropy=0
for i in range(1,len(T)):
    entropy += (T[i]-T[i-1]) * ( (Cp0T001[i]/T[i]) + (Cp0T001[i-1]/T[i-1]) )/4
    S.append(entropy)
    ST.append((T[i]+T[i-1])/2)
aR[0].plot(ST,S, color = 'red')
aR[0].set_ylabel("Entropy (J/K/Ce)",rotation = -90, fontsize = ylabelsize, color = 'red', labelpad = 30)
aR[0].set_ylim(0,6.8)
# aR[0].set_yticks([0,2,4,6])
# aR[0].set_yticklabels([0,2,4,6])
aR[0].plot(T, [8.314*np.log(2)]*210, '-.', color = 'red')
# ax[1,1].plot(T,Cp3T001,label="3T",color="pink")
# ax[1,1].plot(T,Cp4p5T001,label="4.5T",color="purple")
# ax[1,1].plot(T,Cp6T001,label="6T",color="orange")
# ax[1,1].plot(T,Cp7p5T001,label="7.5T",color="black")
# ax[1,1].plot(T,Cp9T001,label="9T",color="cyan")

Cp73mK=np.loadtxt("PRL\Ce_Cp_bri.txt")
S73mK=np.loadtxt("PRL\Ce_entropy_bri.txt")
ax[1,1].plot(Cp73mK[80:,0], Cp73mK[80:,1]/Cp73mK[80:,0],'o',alpha=1,color="blue", label ="Specific heat") #call it 0.3K
aR[0].plot(S73mK[:,0], S73mK[:,1]/2,'s',alpha=1,color="red", label ="Entropy  ")

alpha=0.90
# ax[1,1].plot(Cp0T_001[:,0], alpha*Cp0T_001[:,1]/Cp0T_001[:,0],'o',alpha=1,color="blue", label ="specific heat")
# ax[1,1].plot(Cp3T_001[:,0], alpha*Cp3T_001[:,1],'.',alpha=0.3,color="pink")
# ax[1,1].plot(Cp4p5T_001[:,0], alpha*Cp4p5T_001[:,1],'.',alpha=0.3,color="purple")
# ax[1,1].plot(Cp6T_001[:,0], alpha*Cp6T_001[:,1],'.',alpha=0.3,color="orange")
# ax[1,1].plot(Cp7p5T_001[:,0], alpha*Cp7p5T_001[:,1],'.',alpha=0.3,color="black")
# ax[1,1].plot(Cp9T_001[:,0], alpha*Cp9T_001[:,1],'.',alpha=0.3,color="cyan")

ax[1,1].set_xlim(0,20)
# ax[1,1].set_xscale('log')
ax[1,1].set_ylim(0,1.5)
ax[1,1].set_xlabel("Temperature (K)", fontsize=xlabelsize)
ax[1,1].set_ylabel(r"C$_{\rm mag}$/T (J/mol-FU$\cdot$K/Ce)",fontsize=ylabelsize)
ax[1,1].annotate('$H$ = 0 T', xy=(0.65, 0.59), xycoords='axes fraction',fontsize = annotatesize)
# ax[1,1].annotate('(e)', xy=(0.02, 0.9), xycoords='axes fraction',fontsize = plotlabelsize)
ax[1,1].annotate('Rln(2)', xy=(0.8, 0.87),color='red', xycoords='axes fraction',fontsize = annotatesize)
ax[1,1].legend(loc="center right", frameon = False)
aR[0].legend(loc="center right",bbox_to_anchor=(0.9,0.4), frameon = False)

EX=np.load('PRL\CEF\CEF_E.npy')
Eplot=np.load('PRL\CEF\Eplot.npy')
ax[1,2].plot(Eplot,EX[:,0],label='Simulation',color='blue',zorder=2)
ax[1,2].plot(EX[:,1]/0.00034,EX[:,0],'o',color='red',label='Experiment',zorder=1)
ax[1,2].set_ylabel('Energy (meV)',fontsize=xlabelsize)
ax[1,2].set_xticks([0, 0.1, 0.2])
ax[1,2].set_xlabel('Intensity (arb.unit)',fontsize=ylabelsize)
ax[1,2].legend(loc='upper right',frameon=False,fontsize=17)
# ax[1,2].annotate('(f)', xy=(0.02, 0.9), xycoords='axes fraction',fontsize = plotlabelsize)

# CEFint=np.loadtxt("PRL\CEFint.txt")
CEFint=np.loadtxt("PRL\CEFint40to70.txt")
for i in range(len(CEFint)):
    for j in range(len(CEFint[0])):
        if CEFint[i][j] < -1:
            CEFint[i][j] = np.nan
CEFint = CEFint[0:len(CEFint)//2, :]
CEFint = cv2.resize(CEFint, (162,130))
pos = ax[0,2].imshow(CEFint, origin='lower', cmap='jet', vmin = 0, vmax = 5e-5)
tick_locs=[CEFint.shape[1]*1/6, CEFint.shape[1]*2/6, CEFint.shape[1]*3/6, CEFint.shape[1]*4/6, CEFint.shape[1]*5/6, CEFint.shape[1]*6/6]
tick_lbls=[1, 2, 3, 4, 5, 6]
E_loc = [0,CEFint.shape[0]*20/75, CEFint.shape[0]*40/75, CEFint.shape[0]*60/75]
E_lbls= [0,20, 40, 60]
ax[0,2].set_xticks(tick_locs,tick_lbls)
ax[0,2].set_yticks(E_loc,E_lbls)
ax[0,2].set_xlabel(r'$|\bf Q|$ ($\AA^{-1}$)',fontsize=xlabelsize)
ax[0,2].set_ylabel('Energy (meV)',fontsize=ylabelsize)
# ax[0,2].annotate('(c)', xy=(0.02, 0.9), xycoords='axes fraction',fontsize = plotlabelsize)
cb3=fig.colorbar(pos, ax=ax[0,2],location = 'right',\
                  shrink = 0.85, pad =0.01, aspect = 30,\
                  ticks = [0, 0.000008, 0.000016, 0.000024, 0.000032, 0.00004])
cb3.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize = 20)
cb3.set_label('Intensity (arb.unit)', fontsize = 25)
for i in range(2):
    for j in range(3):
        ax[i,j].tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5)
        for axis in ['top','bottom','left','right']:
            ax[i,j].spines[axis].set_linewidth(3)
plt.savefig(r'PRL\Fig1.png', format='png')
#%%
CEFint=np.loadtxt("PRL\CEFint40to70.txt")
for i in range(len(CEFint)):
    for j in range(len(CEFint[0])):
        if CEFint[i][j] < -1:
            CEFint[i][j] = np.nan
CEFint = CEFint[0:len(CEFint)//2, :]
CEFint = cv2.resize(CEFint, (130,130))
f, (a0, a1) = plt.subplots(1, 2,figsize=(8,6), gridspec_kw={'width_ratios': [2.75, 1]})
pos = a0.imshow(CEFint, origin='lower', cmap='jet', vmin = 0, vmax = 5e-5)
E_loc = [0,CEFint.shape[0]*10/30, CEFint.shape[0]*20/30, CEFint.shape[0]*30/30]
E_lbls= [40,50, 60, 70]
tick_locs=[CEFint.shape[1]*(2-1.8)/3.6, CEFint.shape[1]*(3-1.8)/3.6, CEFint.shape[1]*(4-1.8)/3.6, CEFint.shape[1]*(5-1.8)/3.6]
tick_lbls=[2, 3, 4, 5]
a0.set_xticks(tick_locs,tick_lbls)
a0.set_yticks(E_loc,E_lbls)
a0.set_xlabel(r'$|\bf Q|$ ($\rm \AA ^{-1}$)',fontsize=xlabelsize)
a0.set_ylabel('Energy transfer (meV)',fontsize=ylabelsize)
a0.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5)
for axis in ['top','bottom','left','right']:
    a0.spines[axis].set_linewidth(3)

a1.plot(Eplot,EX[:,0],label='Simulation',color='blue',zorder=2)
a1.scatter(EX[:,1]/0.00034,EX[:,0],marker='o',facecolors='white',edgecolors='red',label='Experiment',zorder=1)
a1.set_ylim(40,70)
a1.set_xlim(0,0.3)
a1.set_yticks([])
a1.set_xticks([0.05,0.15,0.25], [0.2,0.6,1])
# a1.set_xticks([], [])
# a1.set_xlabel(r'Intensity(arb. unit)',fontsize=xlabelsize)
a1.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5)
for axis in ['top','bottom','left','right']:
    a1.spines[axis].set_linewidth(3)
    
f.tight_layout()
f.subplots_adjust(wspace=-0.)
plt.savefig(r'PRL\Fig1f.svg', format='svg',bbox_inches = "tight")

#%%-------------------Fig1d-------------------------------
fig, ax = plt.subplots(figsize=(8,6),dpi = 600)
ax.plot(field,np.array(Total_Magnetization)*0.5,color="blue",zorder=0)
ax.scatter(expM001_1p8K[:,0][::60]/10000,expM001_1p8K[:,1][::60],marker='o',facecolors='white',edgecolors="blue",label=r"$\boldsymbol{\rm H}\parallel$[0,0,1]",zorder=1)
ax.legend(loc="lower right", frameon = False)
ax.plot(field,mag100/4,color="green", zorder=0)
ax.scatter(expM100_1p8K[:,0][::60]/10000,expM100_1p8K[:,1][::60],marker='v',facecolors='white',edgecolors="green",label=r"$\boldsymbol{\rm H}\parallel$[1,0,0]",zorder=1)
ax.legend(loc="lower right", frameon = False)
ax.plot(field,mag110/4,color="purple", zorder=0)
ax.scatter(expM110_1p8K[:,0][::60]/10000,expM110_1p8K[:,1][::60],marker='s',facecolors='white',edgecolors="purple",label=r"$\boldsymbol{\rm H}\parallel$[1,1,0]",zorder=1)
ax.legend(loc="lower right", frameon = False)
ax.set_xlabel("Magnetic Field (T)",fontsize=xlabelsize)
ax.set_ylabel(r'$M$ ($\mu_{\rm B}$/Ce)', fontsize=ylabelsize)
ax.annotate('$T$ = 1.8 K', xy=(0.75, 0.35), xycoords='axes fraction',fontsize = annotatesize)
ax.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
plt.savefig(r'PRL\Fig1d.svg', format='svg',bbox_inches = "tight")
#%%---------------------Fig1e---------------------
fig, ax = plt.subplots(figsize=(8,6), dpi = 600)
aR = np.array([ax.twinx()])
S=[]
ST=[]
entropy=0
for i in range(1,len(T)):
    entropy += (T[i]-T[i-1]) * ( (Cp0T001[i]/T[i]) + (Cp0T001[i-1]/T[i-1]) )/4
    S.append(entropy)
    ST.append((T[i]+T[i-1])/2)
aR[0].plot(ST,S, color = 'red', zorder=1)
aR[0].set_ylabel(r"$\Delta S$ (J/K/mol$_{\rm Ce}$)",rotation = -90, fontsize = ylabelsize, color = 'red', labelpad = 30)
aR[0].set_ylim(0,6.8)
aR[0].plot(T, [8.314*np.log(2)]*210, '-.', color = 'red')


Cp73mK=np.loadtxt("PRL\Ce_Cp_bri.txt")
S73mK=np.loadtxt("PRL\Ce_entropy_bri.txt")
ax.plot(T[0:165],Cp0T001[0:165]/T[0:165]/2,color="blue",zorder=1)
ax.scatter(Cp73mK[80::5,0], Cp73mK[80::5,1]/Cp73mK[80::5,0],facecolors='white',color="blue",label =r"Specific heat", zorder=2) #call it 0.3K
aR[0].scatter(S73mK[::5,0], S73mK[::5,1]/2, marker='s', facecolors='white', edgecolors='red',zorder=2, label =r"Entropy")


ax.set_xlim(0,20)
# ax[1,1].set_xscale('log')
ax.set_ylim(0,1.5)
ax.set_xlabel("Temperature (K)", fontsize=xlabelsize)
ax.set_ylabel(r"$C_{\rm mag}$/$T$ (J/K$^2$/mol$_{\rm Ce}$)",fontsize=ylabelsize)
ax.annotate(r'$\mu_0H$ = 0 T', xy=(0.75, 0.57), xycoords='axes fraction',fontsize = annotatesize)
# ax[1,1].annotate('(e)', xy=(0.02, 0.9), xycoords='axes fraction',fontsize = plotlabelsize)
ax.annotate('Rln(2)', xy=(0.8, 0.87),color='red', xycoords='axes fraction',fontsize = annotatesize)
ax.legend(loc="center right", frameon = False)
aR[0].legend(loc="center right",bbox_to_anchor=(0.92,0.4), frameon = False)

ax.tick_params(width=3,direction='in',length=7,right=True,top=True,pad=5)
for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(3)
plt.savefig(r'PRL\Fig1e.svg', format='svg',bbox_inches = "tight")
#%%#########Along Specific reciprocal space direction
######################################################%%#########################################

def HHL(H):
    #Plot the HHL direction of intensities
    HHL_3=HH0(R2,R3,Jmatpa,H,gmatpa,3)+HH0(R1,R4,Jmatpp,H,gmatpp,3)
    HHL_2=HH0(R2,R3,Jmatpa,H,gmatpa,2)+HH0(R1,R4,Jmatpp,H,gmatpp,2)
    HHL_1=HH0(R2,R3,Jmatpa,H,gmatpa,1)+HH0(R1,R4,Jmatpp,H,gmatpp,1)
    plt.plot(HHL_3[0],HHL_3[1],label=f'{np.round(egy[3],2)} meV')
    plt.plot(HHL_2[0],HHL_2[1],label=f'{np.round(egy[2],2)} meV')
    plt.plot(HHL_1[0],HHL_1[1],label=f'{np.round(egy[1],2)} meV')
    if math.isclose(egy[2], egy[3], rel_tol=1e-2):
        plt.plot(HHL_1[0],HHL_2[1]+HHL_3[1],label='Sum')###########
    elif math.isclose(egy[2], egy[1], rel_tol=1e-2): 
        plt.plot(HHL_1[0],HHL_2[1]+HHL_1[1],label='Sum')###########
    plt.legend(loc='upper right')
    plt.xlabel('HH0')
    plt.ylabel('Intensity(arb.unit)')
    plt.show()
    
def Int_hh(B,hh):
    '''hh=-1 means integrate along H-H, hh=1 means integrate along HH'''
    H=np.array([0,0,B])
    ev,ef,hamM=TTM(Jmatpp,H,gmatpp)
    egy=ev-ev[0]
    print(egy.round(3))
    # hh=-1
    s=-0.2
    f=0.2
    int3=np.array(integral(R2,R3,Jmatpa,H,gmatpa,3, hh, s , f   , 0.02))+np.array(integral(R1,R4,Jmatpp,H,gmatpp,3, hh, s , f   , 0.02))
    plt.plot(int3[0], int3[1],label=f'{np.round(egy[3],2)} meV')
    
    int2=np.array(integral(R2,R3,Jmatpa,H,gmatpa,2, hh, s , f   , 0.02))+np.array(integral(R1,R4,Jmatpp,H,gmatpp,2, hh, s , f   , 0.02))
    plt.plot(int2[0], int2[1],label=f'{np.round(egy[2],2)} meV')
    
    int1=np.array(integral(R2,R3,Jmatpa,H,gmatpa,1, hh, s , f   , 0.02))+np.array(integral(R1,R4,Jmatpp,H,gmatpp,1, hh, s , f   , 0.02))
    plt.plot(int1[0], int1[1],label=f'{np.round(egy[1],2)} meV')
    
    plt.legend(loc='upper right')
    plt.show()
    return int3,int2,int1,egy

#%%----------------------Neutron Data HH-----------------------------

def lorentzian(x,Area,width,pos):
    pi=np.pi
    Y=(Area/pi)*(width/2)/((x-pos)**2+(width/2)**2)
    return Y

def EvsHH(B):
    hh = -1
    int3,int2,int1,egy=Int_hh(B,hh)

    E=np.arange(0,2.0,0.01)#energy
    yspace3=[]
    for i in range(33): #33 corresponds to [0-2] in HH
        yspace3.append(gaussian(E,int3[1][i],0.01, egy[3]))
    yspace3 = np.array(yspace3)
    
    yspace2=[]
    for i in range(33):
        yspace2.append(gaussian(E,int2[1][i],0.01, egy[2]))
    yspace2 = np.array(yspace2)
    
    yspace1=[]
    for i in range(33):
        yspace1.append(gaussian(E,int1[1][i],0.01, egy[1]))
    yspace1 = np.array(yspace1)
    
    yspace_all = np.transpose(yspace1+yspace2+yspace3)
    yspace_resize = cv2.resize(yspace_all, (103, 100))
    yspace_resize = cv2.GaussianBlur(yspace_resize, (5, 5),0)
    return yspace_resize
yspace_resizeE4T = EvsHH(4.25)
# plt.imshow(yspace_resizeE4T, origin = 'lower', vmin =0, vmax = 1000, cmap = 'jet')

dataE4THH=np.loadtxt("PRL/4THH0.2.txt")
for i in range(len(dataE4THH)):
    for j in range(len(dataE4THH[0])):
        if dataE4THH[i][j] < -1:
            dataE4THH[i][j] = np.nan
alldata4THH=np.hstack((np.flip(dataE4THH[0:len(dataE4THH)//2,:], axis = 1), yspace_resizeE4T/2.5e5))
alldata4THH= cv2.resize(alldata4THH, (150,81))
# plt.imshow(alldata4THH, origin = 'lower', vmin =0, vmax =0.007, cmap = 'jet')
# plt.how()
#%%
fig, ax['HH'] = plt.subplots(figsize = (8,8),dpi = 600, constrained_layout=True)
pos1 = ax['HH'].imshow(alldata4THH, origin = 'lower', vmin =0, vmax =0.007, cmap = 'jet')
ax['HH'].annotate(r'$\mu_0H$ = 4 T', xy=(0.78,0.05), xycoords='axes fraction',fontsize = 20, color = 'white')
for i in ["HH"]:
    ax[i].tick_params(width=3,direction='in',length=5,right=True,top=True,labelsize=15,pad=5)
    r=1.5
    tick_locs=[0*r, 25*r, 50*r, 75*r, 100*r]
    tick_lbls=['$-2.0$', '$-1.0$', '$0$', '$1.0$', '$2.0$']
    E_loc = [0,  20,  40,  60, 80]
    E_lbls= [0, 0.5, 1.0, 1.5, 2.0]
    ax[i].set_xticks(tick_locs,tick_lbls, fontsize=25)
    ax[i].set_yticks(E_loc,E_lbls, fontsize=25)
    ax[i].set_ylabel("Energy transfer (meV)", fontsize = 25)
    for axis in ['top','bottom','left','right']:
        ax[i].spines[axis].set_linewidth(3)
    ax[i].set_xlabel('[$h$,$h$,0]', fontsize = 25)
plt.savefig(r'PRL/neutron3.svg', format='svg', bbox_inches = "tight")
#%%
# def data_extract(name):
#     txt=pd.read_csv(name,header=None)[189:314][0].str.split()
#     mat=np.stack(txt.values).astype(np.float64)
#     mat=cv2.flip(mat,1)
#     mat=mat[0:100,10:51]
#     # mat=cv2.resize(mat,(75,100))
#     return mat

# EHH_size=cv2.resize(EHH,(81,100))
# mat1=data_extract(f"C:/Users/qmc/BNZS Sunny/HYS/1.7T.txt") #"HYS/0T.txt"
# # mat1=cv2.flip(mat1,1)
# fig, ax=plt.subplots(1,2,figsize=(10,20),sharey=True, dpi= 250)
# fig.subplots_adjust(wspace=0)

# ax[0].tick_params(axis="both",direction="in", pad=5, which="major", labelsize=20, size=10,width=5)
# ax[0].imshow(mat1,vmin=0,vmax=0.0005,origin="lower",cmap='jet')
# ax[0].set_xticks([0,20],[0.2, 0.5])
# ax[0].set_yticks([25,50,75, 100],[0.5,1.0,1.5,2.0])
# for axis in ['top','bottom','left','right']:
#     ax[0].spines[axis].set_linewidth(3)

# ax[1].tick_params(axis="both",direction="in", pad=5, which="major", labelsize=20, size=10,width=5,left=False,right=True)
# ax[1].imshow(EHH_size[:,0:41],vmin=0,vmax=30,origin="lower",cmap='jet')
# ax[1].set_xticks([20,40],[0.5, 1.0])
# ax[1].set_yticks([25,50,75, 100],[0.5,1.0,1.5,2.0])
# for axis in ['top','bottom','left','right']:
#     ax[1].spines[axis].set_linewidth(3)
# # plt.title(exp)
# plt.show()
#%%#####################plot sum-ed up#######
#Make sure EHH0 is saved before
# plt.imshow(EHH,origin='lower',vmin=0,vmax=100)
# plt.colorbar()
# x_locs=[0,37.5,75,112.5,150]
# x_label=[0,0.5, 1,1.5, 2]
# y_locs=[0,50,100,150]
# y_label=[0,0.5,1.0,1.5]
# plt.xticks(x_locs,x_label)
# plt.yticks(y_locs,y_label)
# plt.title(f'Field = {-0.01+2}T')
#%%#############################################################
#   Function to calculate powder averaged intensity
#   use MPI4py to do parallel computing
################################################################
# def powder(v):
#     Q_powder=[]
#     I_powder=[]
#     h=np.linspace(-2,2,21)
#     k=np.linspace(-2,2,21)
#     l=np.linspace(-2,2,21)
#     counter=0
#     for x in range(len(h)):
#         for y in range(len(k)):
#             for z in range(len(l)):            
#                 Q=np.array([h[x],k[y],l[z]]) @ np.array([b1,b2,b3])
#                 I_powder.append(SQ(h[x],k[y],l[z],v,R3,R6,Jmat,H,g)) #+SQ(h[x],k[y],l[z],v,R11,R14, A, -B,C,D,E)).real)
#                 Q_powder.append(np.linalg.norm(Q))
#                 counter=counter+1
#                 print(counter)
    
#     Q_powder=np.array(Q_powder)
#     I_powder=np.array(I_powder)
#     Q_s=np.sort(Q_powder)
#     I_s=I_powder[np.argsort(Q_powder)]
    
#     Int,Q,_=binned_statistic(Q_s, I_s,statistic='mean', bins=30)
#     Qplot=[]
#     for i in range(len(Q)-1):
#         Qplot.append( (Q[i]+Q[i+1])/2 )
#     plt.plot(Qplot,Int,'.')
#     return Q, Int

# pwd3=powder(3)
# pwd2=powder(2)
# pwd1=powder(1)
