#!/usr/bin/env python
# binary Lennard-Jones (KA-LJ) molecular dyanamic simulation
# initialize positions and velocities and test

import scipy as sp
import numpy as np
from random import choice,random   #for random numbers
# for plotting tools:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
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
#----------------------
global Na
global Nb
global N
Na=800
Nb=200
N=Na+Nb
L = 9.4
# define position and velocity arrays (first Na A-, rest B-part)
x = np.zeros(N,float)
y = np.zeros(N,float)
z = np.zeros(N,float)
vx = np.zeros(N,float)
vy = np.zeros(N,float)
vz = np.zeros(N,float)

# initialize positions:
#-----------------------
'''
 For selecting one option, you may comment 
 out a block of lines with three quotes
 as done here
'''
#
#   * initialize positions from file
#        unpack transposes array to be able to read in as x,y,z
x,y,z=sp.loadtxt('initpos',dtype='float',unpack=True)

#   * uniform randomly distributed positions  
#                            (careful with MD simul. because of overlaps)

# for reproducible random numbers: 
#   put following line before 1st rnd number is drawn) 
sp.random.seed(15) # argument is any integer

x,y,z = sp.random.uniform(low=0.0,high=L,size=(3,N))

#   * on lattice  (careful to not crystallize or separate A and B part)
nsitesx= int(round(pow(N,(1.0/3.0))))
dsitesx = L / float(nsitesx)
for ni in range(nsitesx):
  tmpz = (0.5 + ni)*dsitesx
  for nj in range(nsitesx):
    tmpy = (0.5 + nj)*dsitesx
    for nk in range(nsitesx):
      tmpx = (0.5 + nk)*dsitesx
      i=nk+nj*nsitesx+ni*(nsitesx**2)
      x[i]=tmpx
      y[i]=tmpy
      z[i]=tmpz
#        swap randomly sites of B and A part to not start out separated:
for i in range(Na,N):
   j = sp.random.randint(Na)
   x[i],x[j] = x[j],x[i]
   y[i],y[j] = y[j],y[i]
   z[i],z[j] = z[j],z[i]
# 
# initialize velocities
#-----------------------

# * initialize positions & velocities from file
x,y,z,vx,vy,vz=sp.loadtxt('initposvel',dtype='float',unpack=True)

temperature=0.2
maxwellboltzmannvel(temperature)

# Testing Velocities
#---------------------
# * use savetxt to get plain text for further analysis
#    (not print(...) because print includes []
#    vstack and transpose get data in desired format
sp.savetxt('initvelcheck', (sp.transpose(sp.vstack((vx,vy,vz)))))

# * 2d-scatter plot (for data points)
plt.figure()
plt.scatter(vx[:Na],vz[:Na],s=150,color='blue')
plt.scatter(vx[Na:],vz[Na:],s=70,color='red')
plt.xlabel('$v_x$')
plt.ylabel('$v_z$')
plt.show()

# * 3d-Vpython figure

for i in range(N):
  tx=x[i]
  ty=y[i]
  tz=z[i]
  tvx = vx[i]
  tvy = vy[i]
  tvz = vz[i]
  if i < Na :
    sphere(pos=vector(tx,ty,tz),radius=0.5,color=color.blue)
  else:
    sphere(pos=vector(tx,ty,tz),radius=0.2,color=color.red)
  arrow(pos=vector(tx,ty,tz),axis=vector(tvx,tvy,tvz),color=color.green)
# With middle mouse button zoom in & out; with right mouse button rotate
