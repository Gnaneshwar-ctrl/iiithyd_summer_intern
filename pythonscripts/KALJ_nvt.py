#!/usr/bin/env python
# binary Lennard-Jones (KA-LJ) molecular dyanamic simulation
# NVT stochastic temperature bath
#   determine T_meas(t)

import scipy as sp
import numpy as np
from random import choice,random   #for random numbers
# for plotting tools:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from vpython import *

#-- function to draw for all part. velocities from Maxwell Boltzmann distrib.:
def maxwellboltzmannvel(temp):
  global vx
  global vy
  global vz
  nopart=len(vx)
  sigma=np.sqrt(temp) #sqrt(kT/m)
  vx=np.random.normal(0.0,sigma,nopart)
  vy=np.random.normal(0.0,sigma,nopart)
  vz=np.random.normal(0.0,sigma,nopart)
#  make sure that center of mass does not drift
  vx -= sum(vx)/float(nopart)
  vy -= sum(vy)/float(nopart)
  vz -= sum(vz)/float(nopart)
#  make sure that temperature is exactly wanted temperature
  scalefactor = np.sqrt(3.0*temp*nopart/sum(vx*vx+vy*vy+vz*vz))
  vx *= scalefactor
  vy *= scalefactor
  vz *= scalefactor
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
nMD = 50
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

# for reproducible random numbers:
#   put following line before 1st rnd number is drawn)
sp.random.seed(15) # argument is any integer


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
# *  initialize positions & velocities from file
#     initposvel is config. of well equilibrated system at T=0.5
x,y,z,vx,vy,vz=sp.loadtxt('initposvel',dtype='float',unpack=True)

# to not start out in equilibrium, here T=0.2
nstepBoltz = 10
temperature = 0.2
maxwellboltzmannvel(temperature)
# 
# initialize accelerations
#----------------------------
ax,ay,az = acceleration(x,y,z)

# some analysis/testing
#-----------------------
#  (for other quick analysis see KALJ_nve.py)
temparray = np.zeros(nMD,float)

# time loop
#-------------
for tstep in range(1,nMD+1):
  # update positions
  x += vx*Deltat + 0.5*ax*Deltatto2
  y += vy*Deltat + 0.5*ay*Deltatto2
  z += vz*Deltat + 0.5*az*Deltatto2

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

# T-bath: draw every nstepBoltz steps veloc. from Boltz.distr.
  if (tstep % nstepBoltz) == 0 :
    maxwellboltzmannvel(temperature)

# determine kinetic energy and temperature (KALJ m_i=1 for all i)
  kinE=0.5*sum(vx*vx+vy*vy+vz*vz)/float(N)
  tempmeas = kinE*2.0/3.0
  temparray[tstep-1]=tempmeas

## plot T(t)
tarray = np.arange(Deltat,(nMD+1)*Deltat,Deltat)
plt.rcParams['xtick.labelsize']=11
plt.rcParams['ytick.labelsize']=11
plt.figure()
fax = plt.axes()
fax.set_xlim(0,(nMD+1)*Deltat)
fax.set_xticks(np.arange(0,(nMD+1)*Deltat,nstepBoltz*Deltat))
fax.plot(tarray,temparray,color='black')
fax.scatter(tarray,temparray,s=40,color='blue')
fax.set_xlabel('$t$',fontsize=20)
fax.set_ylabel('$T_\mathrm{meas}$',fontsize=20)
#fax.xaxis.set_major_locator(MultipleLocator(nstepBoltz*Deltat))
fax.xaxis.set_minor_locator(MultipleLocator(Deltat))
#plt.savefig('Toft.eps') #replace plt.show with this line  
plt.show()
