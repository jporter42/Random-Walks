# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:03:25 2019

@author: jxp676
"""

import numpy
import pylab
import random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats 
import scipy.optimize

# defining the number of steps 
n = 1000000


  
#creating two array for containing x and y coordinate 
#of size equals to the number of size and filled up with 0's 
x = numpy.zeros(n)
y = numpy.zeros(n)

# filling the coordinates with random variables
for i in range(1, n):
    val = random.randint(1, 4)
    if val == 1:
        x[i] = x[i - 1] + 1
        y[i] = y[i - 1]
    elif val == 2:
        x[i] = x[i - 1] - 1
        y[i] = y[i - 1]
    elif val == 3:
        x[i] = x[i - 1]
        y[i] = y[i - 1] + 1
    else:
        x[i] = x[i - 1]
        y[i] = y[i - 1] - 1
    

#plotting stuff:
pylab.title("Random Walk ($n = " + str(n) + "$ steps)")
pylab.plot(x, y)
pylab.savefig("rand_walk"+str(n)+".png",bbox_inches="tight",dpi=600)
pylab.show()

dist = []
xdist = []
ydist = []

delta = 1000
for i in range(delta,n,delta):
    pythag = ((x[i] - x[i-delta])**2+(y[i]-y[i-delta])**2)**0.5
    xdist.append(x[i]-x[i-delta])
    ydist.append(y[i]-y[i-delta])
    dist.append(round(pythag))
    

c = numpy.zeros([len(dist),2])
for i in range(0, len(c)):
    c[i, 0] = dist[i]
    c[i, 1] = dist.count(dist[i])
    
c2 = numpy.zeros([len(xdist),2])
for i in range(0, len(c)):
    c[i, 0] = xdist[i]
    c[i, 1] = xdist.count(xdist[i])
    
c3 = numpy.zeros([len(ydist),2])
for i in range(0,len(c)):
    c3[i,0] = ydist[i]
    c3[i,1] = ydist.count(ydist[i])
print(c3)


distance = c[:,0]
count = c[:,1]



def gauss(x, a, x0, sigma):
    return a*numpy.exp(-(x-x0)**2/(2*sigma**2))

guess = [20, 0, 20]
popt, pcov = scipy.optimize.curve_fit(gauss, c3[:, 0], c3[:,1], p0=guess)
xplot = numpy.linspace(-120,120,1000)


def rayleigh(r, a, sigma2):
    return a*r*numpy.exp(-((r**2)/sigma2**2))

guess2 = [20,20]
popt2, pcov2 = scipy.optimize.curve_fit(rayleigh, distance, count, p0=guess2)
rplot = numpy.linspace(-80, 80, 1000)


fig, ax = plt.subplots()
ax.plot(x, y, label="Random Walk ($n = " + str(n) + "$ steps)")
ax.legend()
ax.set_xlim([-500,500])
ax.set_ylim([-500,500])
plt.show()

fig2, ax = plt.subplots()
ax.plot(xplot, rayleigh(rplot, *popt2), 'r')
ax.scatter(distance, count)
ax.set_xlim([0, 130])
plt.show()

fig3, ax = plt.subplots()
ax.plot(xplot, gauss(xplot, *popt), 'r')
ax.scatter(ydist, c3[:,1])
ax.set_xlim([-130, 130])
plt.show()