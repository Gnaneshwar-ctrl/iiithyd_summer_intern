#!/usr/bin/env python
# binary Lennard-Jones (KA-LJ) molecular dyanamic simulation
# NVE velocity Verlet
#   same as KALJ_nve.py but add to
#   determine msd(t)   (logarithmic times)

import scipy as sp
import numpy as np
from random import choice,random   #for random numbers
# for plotting tools:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from vpython import *

#------ function to determine acceleration for each particle: ---
def acceleration(x,y,z):
  global L
  global Ldiv2
  global Na,Nb,N
  global epsAA,epsAB,epsBB
  global sigmaAAto12,sigmaABto12,sigmaBBto12
  global sigmaAAto6,sigmaABto6,sigmaBBto6
  global rcutAAto2,rcutABto2,rcutBBto2

  ax=sp.zeros(N)
  ay=sp.zeros(N)
  az=sp.zeros(N)
# AA interactions
  for i in range(Na-1):
    xi=x[i]
    yi=y[i]
    zi=z[i]
    for j in range(i+1,Na):
      xij=xi-x[j]
      yij=yi-y[j]
      zij=zi-z[j]
      #minimum image convention
      if xij > Ldiv2: xij -= L
      elif xij < - Ldiv2: xij  += L
      if yij > Ldiv2: yij -= L
      elif yij < - Ldiv2: yij  += L
      if zij > Ldiv2: zij -= L
      elif zij < - Ldiv2: zij  += L

      rijto2 = xij*xij + yij*yij + zij*zij
      if(rijto2 < rcutAAto2):
        onedivrijto2 = 1.0/rijto2
        fmagtmp= epsAA*(sigmaAAto12*onedivrijto2**7
                     - 0.5*sigmaAAto6*onedivrijto2**4)
        ax[i] += fmagtmp*xij
        ax[j] -= fmagtmp*xij
        ay[i] += fmagtmp*yij
        ay[j] -= fmagtmp*yij
        az[i] += fmagtmp*zij
        az[j] -= fmagtmp*zij

# AB interactions
  for i in range(Na):
    xi=x[i]
    yi=y[i]
    zi=z[i]
    for j in range(Na,N):
      xij=xi-x[j]
      yij=yi-y[j]
      zij=zi-z[j]
      #minimum image convention
      if xij > Ldiv2: xij -= L
      elif xij < - Ldiv2: xij  += L
      if yij > Ldiv2: yij -= L
      elif yij < - Ldiv2: yij  += L
      if zij > Ldiv2: zij -= L
      elif zij < - Ldiv2: zij  += L

      rijto2 = xij*xij + yij*yij + zij*zij
      if(rijto2 < rcutABto2):
        onedivrijto2 = 1.0/rijto2
        fmagtmp= epsAB*(sigmaABto12*onedivrijto2**7
                     - 0.5*sigmaABto6*onedivrijto2**4)
        ax[i] += fmagtmp*xij
        ax[j] -= fmagtmp*xij
        ay[i] += fmagtmp*yij
        ay[j] -= fmagtmp*yij
        az[i] += fmagtmp*zij
        az[j] -= fmagtmp*zij

# BB interactions
  for i in range(Na,N-1):
    xi=x[i]
    yi=y[i]
    zi=z[i]
    for j in range(i+1,N):
      xij=xi-x[j]
      yij=yi-y[j]
      zij=zi-z[j]
      #minimum image convention
      if xij > Ldiv2: xij -= L
      elif xij < - Ldiv2: xij  += L
      if yij > Ldiv2: yij -= L
      elif yij < - Ldiv2: yij  += L
      if zij > Ldiv2: zij -= L
      elif zij < - Ldiv2: zij  += L

      rijto2 = xij*xij + yij*yij + zij*zij
      if(rijto2 < rcutBBto2):
        onedivrijto2 = 1.0/rijto2
        fmagtmp= epsBB*(sigmaBBto12*onedivrijto2**7
                     - 0.5*sigmaBBto6*onedivrijto2**4)
        ax[i] += fmagtmp*xij
        ax[j] -= fmagtmp*xij
        ay[i] += fmagtmp*yij
        ay[j] -= fmagtmp*yij
        az[i] += fmagtmp*zij
        az[j] -= fmagtmp*zij

  return 48*ax,48*ay,48*az
#----------------------
def potential(x,y,z):
  global L
  global Ldiv2
  global Na,Nb,N
  global epsAA,epsAB,epsBB
  global sigmaAAto12,sigmaABto12,sigmaBBto12
  global sigmaAAto6,sigmaABto6,sigmaBBto6
  global rcutAAto2,rcutABto2,rcutBBto2
  global VcutAA,VcutAB,VcutBB

  Vtot = 0.0
# AA interactions
  for i in range(Na-1):
    xi=x[i]
    yi=y[i]
    zi=z[i]
    for j in range(i+1,Na):
      xij=xi-x[j]
      yij=yi-y[j]
      zij=zi-z[j]
      #minimum image convention
      if xij > Ldiv2: xij -= L
      elif xij < - Ldiv2: xij  += L
      if yij > Ldiv2: yij -= L
      elif yij < - Ldiv2: yij  += L
      if zij > Ldiv2: zij -= L
      elif zij < - Ldiv2: zij  += L

      rijto2 = xij*xij + yij*yij + zij*zij
      if(rijto2 < rcutAAto2):
        onedivrijto2 = 1.0/rijto2
        Vtot += epsAA*(sigmaAAto12*onedivrijto2**6
                     - sigmaAAto6*onedivrijto2**3)-VcutAA

# AB interactions
  for i in range(Na):
    xi=x[i]
    yi=y[i]
    zi=z[i]
    for j in range(Na,N):
      xij=xi-x[j]
      yij=yi-y[j]
      zij=zi-z[j]
      #minimum image convention
      if xij > Ldiv2: xij -= L
      elif xij < - Ldiv2: xij  += L
      if yij > Ldiv2: yij -= L
      elif yij < - Ldiv2: yij  += L
      if zij > Ldiv2: zij -= L
      elif zij < - Ldiv2: zij  += L

      rijto2 = xij*xij + yij*yij + zij*zij
      if(rijto2 < rcutABto2):
        onedivrijto2 = 1.0/rijto2
        Vtot += epsAB*(sigmaABto12*onedivrijto2**6
                     - sigmaABto6*onedivrijto2**3)-VcutAB

# BB interactions
  for i in range(Na,N-1):
    xi=x[i]
    yi=y[i]
    zi=z[i]
    for j in range(i+1,N):
      xij=xi-x[j]
      yij=yi-y[j]
      zij=zi-z[j]
      #minimum image convention
      if xij > Ldiv2: xij -= L
      elif xij < - Ldiv2: xij  += L
      if yij > Ldiv2: yij -= L
      elif yij < - Ldiv2: yij  += L
      if zij > Ldiv2: zij -= L
      elif zij < - Ldiv2: zij  += L

      rijto2 = xij*xij + yij*yij + zij*zij
      if(rijto2 < rcutBBto2):
        onedivrijto2 = 1.0/rijto2
        Vtot += epsBB*(sigmaBBto12*onedivrijto2**6
                     - sigmaBBto6*onedivrijto2**3)-VcutBB

  return 4.0*Vtot/float(N)
#----------------------
global Na
global Nb
global N
Na=800
Nb=200
N=Na+Nb
L = 9.4
Ldiv2 = L/2.0
sigmaAA=1.0
sigmaAB=0.8
sigmaBB=0.88
epsAA=1.0
epsAB=1.5
epsBB=0.5
rcutfactor = 2.5
nMD = 1000
Deltat=0.005
Deltatto2=(Deltat**2)

sigmaAAto12 = sigmaAA**12
sigmaABto12 = sigmaAB**12
sigmaBBto12 = sigmaBB**12
sigmaAAto6 = sigmaAA**6
sigmaABto6 = sigmaAB**6
sigmaBBto6 = sigmaBB**6
rcutAAto2 = (rcutfactor*sigmaAA)**2
rcutABto2 = (rcutfactor*sigmaAB)**2
rcutBBto2 = (rcutfactor*sigmaBB)**2

VcutAA = epsAA*(sigmaAAto12/(rcutAAto2**6)-sigmaAAto6/(rcutAAto2**3))
VcutAB = epsAB*(sigmaABto12/(rcutABto2**6)-sigmaABto6/(rcutABto2**3))
VcutBB = epsBB*(sigmaBBto12/(rcutBBto2**6)-sigmaBBto6/(rcutBBto2**3))

# for logarithmic printing of msd
nstepBoltz = nMD+1
kmsdmax = 60
t0msd = 1.0
A = (float(nMD)/t0msd)**(1.0/float(kmsdmax))
tmsd = t0msd
tmsdnextint = int(round(t0msd))

# define position and velocity arrays (first Na A-, rest B-part)
x = np.zeros(N,float)
y = np.zeros(N,float)
z = np.zeros(N,float)
vx = np.zeros(N,float)
vy = np.zeros(N,float)
vz = np.zeros(N,float)

# 
# initialize positions and velocities
#------------------------------------
# (for other options of initial configurations see  KALJ_initposvel.py)
# * initialize positions & velocities from file
#     initposvel is config. of well equilibrated system at T=0.5
x,y,z,vx,vy,vz=sp.loadtxt('initposvel',dtype='float',unpack=True)
# 
# initialize accelerations
#----------------------------
ax,ay,az = acceleration(x,y,z)
# check accelerations
sp.savetxt('initacccheck', (sp.transpose(sp.vstack((ax,ay,az)))))

# for msd:  store positions at t=0
#-----------------------
x0 = np.copy(x)
y0 = np.copy(y)
z0 = np.copy(z)
xu = np.copy(x)
yu = np.copy(y)
zu = np.copy(z)
fileoutmsd= open("msd.data",mode='w')

# for plotting y_i(t)
yiplotarray =  np.zeros(nMD,float)
kinEarray =  np.zeros(nMD,float)
Vpotarray =  np.zeros(nMD,float)

'''
# t=0 snapshot for animation
s = np.empty(N,sphere)
ar = np.empty(N,arrow)
for i in range(N):
  if i < Na :
    s[i] = sphere(pos=vector(x[i],y[i],z[i]),radius=0.5,color=color.blue)
  else:
    s[i] = sphere(pos=vector(x[i],y[i],z[i]),radius=0.2,color=color.red)
  ar[i] = arrow(pos=vector(x[i],y[i],z[i]), axis=vector(vx[i],vy[i],vz[i]),color = color.green)
'''

# time loop
#-------------
for tstep in range(1,nMD+1):
  # update positions
  x += vx*Deltat + 0.5*ax*Deltatto2
  y += vy*Deltat + 0.5*ay*Deltatto2
  z += vz*Deltat + 0.5*az*Deltatto2
  xu += vx*Deltat + 0.5*ax*Deltatto2  #for msd
  yu += vy*Deltat + 0.5*ay*Deltatto2  #for msd
  zu += vz*Deltat + 0.5*az*Deltatto2  #for msd


  # periodic boundary conditions:
  for i in range(N):
    if x[i] > L: x[i] -= L
    elif x[i] <= 0:  x[i] += L
    if y[i] > L: y[i] -= L
    elif y[i] <= 0:  y[i] += L
    if z[i] > L: z[i] -= L
    elif z[i] <= 0:  z[i] += L

  # update velocities
  vx += 0.5*ax*Deltat
  vy += 0.5*ay*Deltat
  vz += 0.5*az*Deltat
  ax,ay,az = acceleration(x,y,z)
  vx += 0.5*ax*Deltat
  vy += 0.5*ay*Deltat
  vz += 0.5*az*Deltat

  # for testing with y_8(t) (8th part., similarly x,z,& other part.)
  yiplotarray[tstep-1]=y[7]

  #print ((tstep-0.5)*Deltat, potential(x,y,z))
  # determine kinetic energy per particle
  kinE=0.5*sum(vx*vx+vy*vy+vz*vz)/float(N)
  kinEarray[tstep-1]=kinE

  # determine potential energy per particle
  Vpot = potential(x,y,z)
  Vpotarray[tstep-1]=Vpot

  '''
  # for animation (takes long time, so comment out if figures below wanted)
  rate(30)
  for i in range(N):
    s[i].pos = vector(x[i],y[i],z[i])
    ar[i].pos = vector(x[i],y[i],z[i])
    ar[i].axis = vector(vx[i],vy[i],vz[i])
  '''
  # measure msd:
  #  check whether tstep has reached next time measurement (for log-times)
  if tmsdnextint == tstep:
    # prepare when next msd-time
    while(tmsdnextint == tstep):
      tmsd = A*tmsd
      tmsdnextint = int(round(tmsd))
    # do measurement
    msdA = 0.0
    for i in range(Na):
      dx = xu[i]-x0[i]
      dy = yu[i]-y0[i]
      dz = zu[i]-z0[i]
      msdA += dx*dx+dy*dy+dz*dz
    msdA /= float(Na)

    msdB = 0.0
    for i in range(Na,N):
      dx = xu[i]-x0[i]
      dy = yu[i]-y0[i]
      dz = zu[i]-z0[i]
      msdB += dx*dx+dy*dy+dz*dz
    msdB /= float(Nb)
    # print into file
    print(tstep*Deltat,msdA,msdB,file=fileoutmsd)

'''
## 2d-scatter plot positions z(x) for t=0 and after 50 MD steps
plt.figure()
plt.scatter(x,z,s=30,color='red')
plt.scatter(x0,z0,s=15,color='black')
plt.xlim(0,L)
plt.xlabel('$x$')
plt.ylabel('$z$')
plt.show()

# plot y_8(t)
tarray = np.arange(Deltat,(nMD+1)*Deltat,Deltat)
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.figure()
plt.plot(tarray,yiplotarray,color='black')
plt.xlabel('$t$',fontsize=15)
plt.ylabel('$y_8$',fontsize=15)
#plt.savefig('y8oft.eps') #for eps-file replace plt.show with this line
plt.show()

# plot kinetic energy K(t)/N
tarray = np.arange(Deltat,(nMD+1)*Deltat,Deltat)
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.figure()
plt.plot(tarray,kinEarray,color='black')
plt.xlabel('$t$',fontsize=15)
plt.subplots_adjust(left=0.15)
plt.ylabel('$E_{kin}/N$',fontsize=15)
#plt.savefig('Ekinoft.eps') #for eps-file replace plt.show with this line
plt.show()

# plot E_Vpot(t)/N
tarray = np.arange(Deltat,(nMD+1)*Deltat,Deltat)
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.figure()
plt.plot(tarray,Vpotarray,color='black')
plt.xlabel('$t$',fontsize=15)
plt.subplots_adjust(left=0.15)
plt.ylabel('$V/N$',fontsize=15)
#plt.savefig('Epotoft.eps') #for eps-file replace plt.show with this line
plt.show()

# plot E_tot(t)/N
tarray = np.arange(Deltat,(nMD+1)*Deltat,Deltat)
#plt.rcParams['xtick.labelsize']=11
#plt.rcParams['ytick.labelsize']=11
plt.figure()
plt.plot(tarray,kinEarray+Vpotarray,color='black')
plt.xlabel('$t$',fontsize=15)
plt.subplots_adjust(left=0.2)
plt.ylabel('$E_{tot}/N$',fontsize=15)
#plt.savefig('Etotoft.eps') #for eps-file replace plt.show with this line
plt.show()
'''
