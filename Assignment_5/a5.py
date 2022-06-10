"""
        EE2703 Applied Programming Lab - 2022
        Assignment 5: The Resistor Problem
        Done by: Ishaan Agarwal
        Roll Number: EE20B046
        Date: 7th February, 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import pylab as pl
import argparse

#accepting command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--Nx", default = 25, help="size along x")
parser.add_argument("--Ny", default = 25, help="size along y")
parser.add_argument("--radius", default = 8, help="radius of central lead")
parser.add_argument("--Niter", default = 1500, help="number of iterations")
vars = parser.parse_args()

#assigning variables
Nx = int(vars.Nx)
Ny = int(vars.Ny)
radius = float(vars.radius)
Niter = int(vars.Niter)

#creating the potential grid
phi = np.zeros((Ny,Nx))

#creating position vectors
x = np.linspace(-((Nx-1)//2), Nx//2, Nx) # since phi uses integral indices, we cannot just do -Nx/2,Nx/2
y = np.linspace(-((Ny-1)//2), Ny//2, Ny) 

X,Y = np.meshgrid(x,y)

#using numpy.where() to find the points within the radius
ii = np.where(X**2 + Y**2 <= radius**2)

#set potential at those points as one
phi[ii] = 1

#contour plot of phi versus x and y
plt.contourf(x, y, phi)
plt.colorbar()
plt.scatter(X[ii], Y[ii], c='r', marker = 'o', label = 'points within radius (V=1V)')
plt.plot(0,0, 'bo', label = 'center')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of Potential')
plt.legend()
plt.show()

def update_phi(phi, oldphi): #update phi using given formula
    phi[1:-1, 1:-1] = (oldphi[0:-2, 1:-1] + oldphi[2:, 1:-1] + oldphi[1:-1, 0:-2] + oldphi[1:-1, 2:])/4
    return phi

def boundary_conditions(phi): #boundary conditions
    phi[1:-1, 0] = phi[1:-1, 1]
    phi[1:-1, -1] = phi[1:-1, -2]
    phi[0, 1:-1] = phi[1, 1:-1]
    phi[-1, 1:-1] = 0
    phi[ii] = 1
    return phi


#keeping track of errors
errors = np.zeros(Niter)

for k in range(Niter):
    oldphi = np.copy(phi)   #making a copy of phi
    phi = update_phi(phi, oldphi) #updating phi
    phi = boundary_conditions(phi) #applying boundary conditions
    errors[k] = (np.abs(phi - oldphi).max()) #appending error

#plotting the errors
plt.plot(range(Niter)[::50], errors[::50], 'ro--', label = 'error')
plt.xlabel('iteration')
plt.ylabel('error')
plt.title('Error vs. iteration')
plt.legend()
plt.show()

#plotting the errors of every 50th iteration on a semilogy plot
plt.semilogy(range(Niter)[::50], errors[::50], 'ro--', label = 'error')
plt.xlabel('iteration')
plt.ylabel('error')
plt.title('Error vs iteration on a semilogy scale')
plt.legend()
plt.show()

#plotting the errors on a log log plot
plt.loglog(range(Niter)[::50], errors[::50], 'ro--', label = 'error')
plt.xlabel('iteration')
plt.ylabel('error')
plt.title('Error vs iteration on a loglog scale')
plt.legend()
plt.show()

#finding the least squares fit using lstsq
def error_fit(x, y):
    a = np.vstack([x, np.ones(len(x))]).T
    B, logA = np.linalg.lstsq(a, y)[0]
    return B, np.exp(logA)

#fitting an exponential
def exp_fit(x, A, B):
    return A*np.exp(B*x)

errors[errors == 0] = np.min(errors[errors != 0])*10**(-20) #ensuring there are no zero values before taking log

#finding fit 1
x1 = range(Niter)
y1 = np.log(errors)
B1, A1 = error_fit(x1, y1)
plt.semilogy(x1[::50], exp_fit(x1[::50], A1, B1), 'ro--', label = 'Fit1', ms = 5, alpha = 0.5)

#finding fit 2
x2 = range(500, Niter)
y2 = np.log(errors[500:])
B2, A2 = error_fit(x2, y2)
plt.semilogy(x2[::50], exp_fit(x2[::50], A2, B2), 'bo--', label = 'Fit2', ms = 5, alpha = 0.5)

#original
plt.semilogy(range(Niter)[::50], errors[::50], 'ko--', label = 'Original errors', ms = 3, linewidth = 3, alpha = 0.8)
plt.title('Fitting using least squares')
plt.legend()
plt.show()

#upper bound of errors
def max_error(A, B, N):
    return -A*(np.exp(B*(N+0.5)))/B

#plotting the upper bounds of the errors
plt.semilogy(range(Niter)[::50], max_error(A1, B1, np.array(range(Niter)[::50])), 'ro--', label = 'Upper bound of errors')
plt.title('Upper bound of errors')
plt.legend()
plt.show()

#surface plot of potential
fig = plt.figure()
ax = p3.Axes3D(fig)
surf_plot = ax.plot_surface(X, Y, phi, rstride = 1, cstride = 1, cmap = 'jet')
fig.colorbar(surf_plot)
ax.set_xlabel('y')
ax.set_ylabel('x')
ax.set_zlabel('potential')
ax.set_title('Surface plot of Potential')
plt.show()

#contourf plot of potential
plt.contourf(x, -y, phi)
plt.colorbar()
plt.scatter(X[ii], Y[ii], c = 'r', marker = 'o', label = 'points within radius (V=1V)')
plt.plot(0,0, 'bo', label = 'center')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Contour plot of Potential')
plt.legend()
plt.show()

#vector plot of currents
#creating jx and jy of same dimensions as phi
jx = np.zeros(phi.shape)
jy = np.zeros(phi.shape)

jx[:, 1:-1] = (phi[:, 0:-2] - phi[:, 2:])/2
jy[1:-1, :] = (- phi[0:-2, :] + phi[2:, :])/2

#plotting vector plot using quiver
plt.quiver(x, y, jx[::-1,:], jy[::-1,:], scale = 4)
plt.scatter(X[ii], Y[ii], c = 'r', marker = 'o', label = 'points within radius (V=1V)')
plt.plot(0,0, 'bo', label = 'center')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector plot of currents')
plt.legend()
plt.show()





















