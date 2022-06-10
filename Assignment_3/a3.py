"""
        EE2703 Applied Programming Lab - 2022
        Assignment 3: Modified Nodal Analysis
        Done by: Ishaan Agarwal
        Roll Number: EE20B046
        Date: 7th February, 2022
"""

from statistics import stdev
import matplotlib.pyplot as plt
from pylab import *
import scipy.special as sp
import numpy as np

# Question 2

arr = np.loadtxt("fitting.dat")
time = np.loadtxt("fitting.dat", usecols=0)
data = np.loadtxt("fitting.dat", usecols=range(1,10))


#Question 3

#plotting the noise
noise = []
for i in range(len(data[0])):
    noise.append(data[:,i] - 1.05*sp.jn(2, time)+0.105*time)
    plt.plot(time, noise[i])
# print(shape(noise)) 9*101
plt.title("Noise for all 9 datasets")
plt.xlabel("Time")
plt.ylabel("Noise")
plt.legend(["σ1 = 0.100", "σ2 = 0.056", "σ3 = 0.032", "σ4 = 0.018", "σ5 = 0.010", "σ6 = 0.006", "σ7 = 0.003", "σ8 = 0.002", "σ9 = 0.001"])
plt.axis([0, 10, -0.3, 0.3])
plt.grid()
plt.show()

#plotting the probability distribution
noise_dist = []
sigma = logspace(-1, -3, 9)
j=0

for i in sigma:
    noise_dist.append(np.exp(-(noise[j]**2)/(2*i**2))/(i*np.sqrt(2*np.pi)))
    plt.scatter(noise[j], noise_dist[j])
    j+=1
plt.title("Probability distribution for all 9 datasets")
plt.xlabel("Noise")
plt.ylabel("Probability distribution Function")
plt.legend(["σ1 = 0.100", "σ2 = 0.056", "σ3 = 0.032", "σ4 = 0.018", "σ5 = 0.010", "σ6 = 0.006", "σ7 = 0.003", "σ8 = 0.002", "σ9 = 0.001"])
plt.grid()
plt.show()


#Question 4

#Creating Function to fit the data
def g(t, A, B):
    return A*sp.jn(2, t) + B*t

true_A = 1.05
true_B = -0.105

#plotting the data
for i in range(len(data[0])):
    plt.plot(time, data[:,i])
plt.plot(time, g(time, 1.05, -0.105), 'k')
plt.title("Data")
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend(["σ1 = 0.100", "σ2 = 0.056", "σ3 = 0.032", "σ4 = 0.018", "σ5 = 0.010", "σ6 = 0.006", "σ7 = 0.003", "σ8 = 0.002", "σ9 = 0.001", "True value"])
plt.grid()  
plt.show()


#Question 5

#Plotting with error bars
plt.errorbar(time[::5], data[::5,0], yerr=0.1, fmt='ro')
plt.plot(time, g(time, 1.05, -0.105), 'k')
plt.title("Data with error bars")
plt.xlabel("Time")
plt.ylabel("Data")
plt.legend(["σ1 = 0.100", "True value"])
plt.grid()
plt.show()


#Question 6

#Creating Matrix Equation and verifying the solution
x = sp.jn(2, time)
M = c_[x, time]
p = np.array([1.05, -0.105])
if np.allclose(np.matmul(M, p),g(time, 1.05, -0.105)):
    print("Matrix Equation is correct \n")
else:
    print("Matrix Equation is incorrect \n")


#Question 7

#Computing mean squared errors
A = np.linspace(0, 2, 21)
B = np.linspace(-0.2, 0, 21)
e = np.zeros((len(A), len(B)))
for i in range(len(A)):
    for j in range(len(B)):
        e[i,j] = np.mean((g(time, A[i], B[j]) - data[:,0])**2)
#print(e)
#print("\n")


#Question 8

#Plotting a contour plot of e
X, Y = np.meshgrid(A, B) #creating a grid
plt.contour(X, Y, e)
plt.clabel(plt.contour(X, Y, e), fontsize=10) #labelling the contour plot
plt.plot(true_A, true_B, 'ro')
plt.annotate("True Value", (true_A, true_B))
plt.title("Contour plot of e")
plt.xlabel("A")
plt.ylabel("B")
plt.grid()
plt.show()


#Question 9

#To obtain the best estimate of A and B using lstsq
Best = np.linalg.lstsq(M, data)[0]
A_best = Best[0]
B_best = Best[1]
# A_best = A[np.unravel_index(e.argmin(), e.shape)[0]] #finding the index of the minimum value of e, then finding the coordinates and then the value of A there  (A_best)
# B_best = B[np.unravel_index(e.argmin(), e.shape)[1]] #finding the index of the minimum value of e, then finding the coordinates and then the value of B there  (B_best)
#print(f"A_best = {A_best} \nB_best = {B_best}")


#Question 10

#Plotting the errors for different datasets

error_A_best = abs(A_best - true_A) #finding the absolute value of the difference between the best estimate of A and the true value of A
error_B_best = abs(B_best - true_B) #finding the absolute value of the difference between the best estimate of B and the true value of B

plt.plot(sigma, error_A_best, 'ro', linestyle = '--')
plt.plot(sigma, error_B_best, 'go', linestyle = '--')
plt.legend(["error in A", "error in B"])
plt.title("Error in A and B")
plt.xlabel("sigma")
plt.ylabel("error")
plt.grid()
plt.show()


#Question 11

#Plotting loglog curves for the errors vs noise
plt.errorbar(sigma, error_A_best, yerr = sigma, fmt='ro') #error bars for A
plt.errorbar(sigma, error_B_best, yerr = sigma, fmt='go') #error bars for B
plt.axis([1e-3, .1, 1e-5, 0.1]) 
plt.xscale('log') 
plt.yscale('log')
plt.legend(["error in A", "error in B"])
plt.title("Error in A and B vs sigma")
plt.xlabel("sigma")
plt.ylabel("error")
plt.grid()
plt.show()

















