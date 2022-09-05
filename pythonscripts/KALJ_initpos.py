#!/usr/bin/env python
# binary Lennard-Jones (KA-LJ) molecular dyanamic simulation
# initialize positions and test

import scipy as sp
import numpy as np
from random import choice,random   #for random numbers
# for plotting tools:
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import matplotlib.pyplot as plt
from vpython import *

global Na
global Nb
global N
Na=800
Nb=200
N=Na+Nb
L = 9.4
# define position arrays (first Na A-, rest B-part)
x = np.zeros(N,float)
y = np.zeros(N,float)
z = np.zeros(N,float)

# initialize positions:
#-----------------------
# For selecting one option, you may comment out a block of lines with 
#        '''   textblock     '''
#
#   * initialize positions from file
#        unpack transposes array to be able to read in as x,y,z
x,y,z=sp.loadtxt('initpos',dtype='float',unpack=True)

#   * uniform randomly distributed positions  
#                            (careful with MD simul. because of overlaps)
sp.random.seed(15) # argument is any integer  (for reproducible random numbers)  This line should be in program before first random number is drawn.
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
#    end of initialization of positions on lattice

# Testing: 
#---------
# * use savetxt to get plain text for further analysis
#    (not print(...) because print includes []
#    vstack and transpose get data in desired format
sp.savetxt('initposcheck', (sp.transpose(sp.vstack((x,y,z)))))

# Plotting:  (info: Newman Ch3 & matplotlib.org)
# * 2d-scatter plot (for data points)
plt.figure()
plt.scatter(x[:Na],z[:Na],s=150,color='blue')
plt.scatter(x[Na:],z[Na:],s=70,color='red')
plt.xlim(0,L)
plt.xlabel('$x$')
plt.ylabel('$z$')
plt.show()

# * 3d-scatter plot
# info: https://matplotlib.org/3.1.1/gallery/mplot3d/scatter3d.html
fig3d = plt.figure()
fax = fig3d.add_subplot(111, projection='3d')
fax.scatter(x[:Na],y[:Na],z[:Na], marker="o",s=150,facecolor='blue')
fax.scatter(x[Na:],y[Na:],z[Na:], marker="o",s=70,facecolor='red')
plt.xlim(0,L)
plt.ylim(0,L)
fax.set_xlabel('$x$')
fax.set_ylabel('$y$')
fax.set_zlabel('$z$')
plt.show()
# With right mouse button zoom in and out; with left mouse button rotate

# * 3d-Vpython figure

for i in range(N):
  tx=x[i]
  ty=y[i]
  tz=z[i]
  if i < Na :
    sphere(pos=vector(tx,ty,tz),radius=0.5,color=color.blue)
  else: 
    sphere(pos=vector(tx,ty,tz),radius=0.2,color=color.red)
# With middle mouse button zoom in & out; with right mouse button rotate
