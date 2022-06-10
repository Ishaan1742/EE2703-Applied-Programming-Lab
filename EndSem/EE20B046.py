# %%
"""
        EE2703 Applied Programming Lab - 2022
        End Semester Examination
        Done by: Ishaan Agarwal
        Roll Number: EE20B046
        Date: 12th May, 2022
"""

# %%
import numpy as np
import matplotlib.pyplot as plt


# %%
#defining the constants

#independent parameters
l = 0.5 #quarter wavelength
c = 2.9979e8 #speed of light
mu0 = 4*np.pi*1e-7 #permeability of free space
N = 4 #Number of sections in each half section of the antenna
Im = 1 #current injected into the antenna
a = 0.01 #radius of the wire

#dependent parameters
lamda = l*4 #wavelength 
f = c/lamda #frequency
k = 2*np.pi/lamda #wave number
dz = l/N #spacing of the elements



# %%
#Question 1
z = np.zeros(2*N+1)
z = np.linspace(-l, l, 2 * N + 1) #creating the array of z and dropping certain values to obtain the array of u
#drop first and last element and middle element of u (known values)
u = np.delete(z, [0, N, -1])

#constructing current vectors (theoretical)
I = Im * np.sin((2 * np.pi / lamda) * (l - abs(z)))  # current vector
I[N] = Im #current injected into the middle element
I[0] = 0 #boundary condition
I[-1] = 0 #boundary condition
#form J by deleting first, last and middle element of I
J = np.delete(I, [0, N, -1])




# %%
#Question 2
#creating M matrix which is 1/(2*pi*a) * Identity matrix (dimension = 2*N-2)
def compute_M(N, a):
    M = np.identity(2*N-2)*(1/(2*np.pi*a))
    return M
M = compute_M(N, a)


# %%
#Question 3
#computing Rz and Ru
#Rz computes distances including distances to known currents whereas Ru computes distances for only unknown currents
def compute_Rz(z, z_dash):
    return np.sqrt((z-z_dash)**2 + a**2)
def compute_Ru(u, u_dash):
    return np.sqrt((u-u_dash)**2 + a**2)

Rz = compute_Rz(u, z.reshape(-1,1))
Ru = compute_Ru(u, u.reshape(-1,1))

# %%
#computing P and Pb
def compute_P(Ru):
    return (mu0/(4*np.pi)) * np.exp(-1j*k*Ru) * (1/Ru) * dz
def compute_Pb(RiN):
    return (mu0/(4*np.pi)) * np.exp(-1j*k*RiN) * (1/RiN) * dz

P = compute_P(Ru)
Pb = compute_Pb(Rz[N,:])


# %%
#Question 4
#computing Qij and Qb
def compute_Qij(Ru, P):
    return -P * (a / mu0) * (complex(0, -k) / Ru - 1 / Ru**2)
def compute_QB(Pb, RiN):
    return -Pb * a / mu0 * ((-1j * k) / RiN - 1 / (RiN**2))

Qij = compute_Qij(Ru, P)
Qb = compute_QB(Pb, Rz[N,:])
Qb = Qb.reshape(-1,1)

# %%
#Question 5
#finding J_calculated and I_calculated
J_calculated = (np.linalg.inv(M-Qij).dot(Qb*Im)).T[0] #obtained was an array of array, thus taking the first element of the array
I_calculated = np.concatenate(([0], J_calculated[:N-1], [Im], J_calculated[N-1:], [0])) 




# %%
#plotting I and I_calculated on the same graph
plt.plot(z, I, label = 'I(assuming sine wave)')
plt.plot(z, I_calculated, label = 'I_calculated')
plt.title('Calculated current vs Assumed current Plot')
plt.grid()
plt.xlabel('z')
plt.ylabel('I')
plt.legend(loc = 'upper right')
plt.show()


# %%
#printing all values for N=4
print('\n')
print('N = 4')
print('\n')
print('z = ', z.round(2))
print('\n')
print('u = ', u.round(2))
print('\n')
print('M = ', M.round(2))
print('\n')
print('Rz = ', Rz.round(2))
print('\n')
print('Ru = ', Ru.round(2))
print('\n')
print('RiN = ', Rz[N,:].round(2))
print('\n')
print('P = ', (P*1e8).round(2))
print('\n')
print('Pb = ', (Pb*1e8).round(2))
print('\n')
print('Qij = ', Qij.round(2))
print('\n')
print('Qb = ', Qb.round(2))
print('\n')
print('J_calculated = ', J_calculated.round(6))
print('\n')
print('I_calculated = ', I_calculated.round(6))
print('\n')




