"""
        EE2703 Applied Programming Lab - 2022
        Assignment 4: Fourier Approximations
        Done by: Ishaan Agarwal
        Roll Number: EE20B046
        Date: 7th February, 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

#Question 1

#defining exponential function
def exp(x):
    return np.exp(x)

#defining cos(cos(x)) function
def cos_cos(x):
    return np.cos(np.cos(x))

#plotting the functions over (-2*pi, 4*pi)
x = np.linspace(-2*np.pi, 4*np.pi, 600)
x1 = np.linspace(-2*np.pi, 0, 200)
x2 = np.linspace(0, 2*np.pi, 200)
x3 = np.linspace(2*np.pi, 4*np.pi, 200)
y_true = exp(x) #true y
y = np.zeros(len(x))
y[0:200] = exp(x2) #creating periodic extension
y[200:400] = exp(x2)
y[400:600] = exp(x2)
z = cos_cos(x)

plt.semilogy(x, y_true, label='True Value of exp(x)', color='red')
plt.semilogy(x, y, label='Periodic extended Value of exp(x)', color='blue')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Plot of exp(x)')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, z, label='cos(cos(x))')
plt.xlabel('x')
plt.ylabel('z')
plt.title('Plot of cos(cos(x))')
plt.legend()
plt.grid()
plt.show()

#Question 2

#dictionary mapping the two functions exp(x) and cos(cos(x))
dict = {'exp(x)':exp, 'cos(cos(x))':cos_cos}


#defining u(x, k) and v(x, k) functions for exp(x)
def u(x, k, func):
    return dict[func](x)*np.cos(k*x)
def v(x, k, func):
    return dict[func](x)*np.sin(k*x)


#calculating fourier coefficients for exp(x)
def a(k, func):
    return integrate.quad(u, 0, 2*np.pi, args = (k, func))[0]/(np.pi)

def b(k, func):
    return integrate.quad(v, 0, 2*np.pi, args = (k, func))[0]/(np.pi)

def a0(func):
    return integrate.quad(u, 0, 2*np.pi, args = (0, func))[0]/(2*np.pi)
    
#Question 3

#storing all coefficients of exp(x) in one list
coeff_exp = []
coeff_exp.append(a0('exp(x)'))
for k in range(1, 26):
    coeff_exp.append(a(k, 'exp(x)'))
    coeff_exp.append(b(k, 'exp(x)'))

#storing all coefficients of cos(cos(x)) in one list
coeff_coscos = []
coeff_coscos.append(a0('cos(cos(x))'))
for k in range(1, 26):
    coeff_coscos.append(a(k, 'cos(cos(x))'))
    coeff_coscos.append(b(k, 'cos(cos(x))'))

#plotting the coefficients vs n for exp(x)
plt.semilogy(range(51), np.abs(coeff_exp), 'ro', label='coefficients for exp(x)')
plt.title('Semilogy Plot of coefficients vs n for exp(x)')
plt.xlabel('n')
plt.ylabel('coefficients')
plt.legend()
plt.grid()
plt.show()

plt.loglog(range(51), np.abs(coeff_exp), 'ro', label='coefficients for exp(x)')
plt.title('Loglog Plot of coefficients vs n for exp(x)')
plt.xlabel('n')
plt.ylabel('coefficients')
plt.legend()
plt.grid()
plt.show()

#plotting the coefficients vs n for cos(cos(x))
plt.semilogy(range(51), np.abs(coeff_coscos), 'ro', label='coefficients for cos(cos(x))')
plt.title('Semilogy Plot of coefficients vs n for cos(cos(x))')
plt.xlabel('n')
plt.ylabel('coefficients')
plt.legend()
plt.grid()
plt.show()

plt.loglog(range(51), np.abs(coeff_coscos), 'ro', label='coefficients for cos(cos(x))')
plt.title('Loglog Plot of coefficients vs n for cos(cos(x))')
plt.xlabel('n')
plt.ylabel('coefficients')
plt.legend()
plt.grid()
plt.show()

#Question 4 and 5

#using least squares method to calculate fourier coefficients

x = np.linspace(0, 2*np.pi, 401)
x = x[:-1] #drop last element to have proper periodic integral

b1 = exp(x)
A = np.zeros((400, 51))
A[:,0] = 1
for k in range(1, 26):
    A[:, 2*k-1] = np.cos(k*x)
    A[:, 2*k] = np.sin(k*x)
#finding the least squares solution c1
c1 = np.linalg.lstsq(A, b1)[0] #best fit vector

b2 = cos_cos(x)
A = np.zeros((400, 51))
A[:,0] = 1
for k in range(1, 26):
    A[:, 2*k-1] = np.cos(k*x)
    A[:, 2*k] = np.sin(k*x)
#finding the least squares solution c2
c2 = np.linalg.lstsq(A, b2)[0] #best fit vector

#plotting the new obtained coefficients along with the true values
plt.semilogy(range(51), np.abs(c1), 'ro', label='Estimated coefficients for exp(x)', alpha = 0.5)
plt.semilogy(range(51), np.abs(coeff_exp), 'go', label='True coefficients for exp(x))', alpha = 0.5)
plt.title('Semilogy Plot of estimated coefficients and true coefficients vs n for exp(x)')
plt.xlabel('n')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

plt.loglog(range(51), np.abs(c1), 'ro', label='Estimated coefficients for exp(x)', alpha = 0.5)
plt.loglog(range(51), np.abs(coeff_exp), 'go', label='True coefficients for exp(x),', alpha = 0.5)
plt.title('Loglog Plot of estimated coefficients and true coefficients vs n for exp(x)')
plt.xlabel('n')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

plt.semilogy(range(51), np.abs(c2), 'ro', label='Estimated coefficients for cos(cos(x))', alpha = 0.5)
plt.semilogy(range(51), np.abs(coeff_coscos), 'go', label='True coefficients for cos(cos(x))', alpha = 0.5)
plt.title('Semilogy Plot of estimated coefficients and true coefficients vs n for cos(cos(x))')
plt.xlabel('n')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

plt.loglog(range(51), np.abs(c2), 'ro', label='Estimated coefficients for cos(cos(x))', alpha = 0.5)
plt.loglog(range(51), np.abs(coeff_coscos), 'go', label='True coefficients for cos(cos(x))', alpha = 0.5)
plt.title('Loglog Plot of estimated coefficients and true coefficients vs n for cos(cos(x))')
plt.xlabel('n')
plt.ylabel('Coefficients')
plt.legend()
plt.grid()
plt.show()

#Question 6

#finding the absolute error between estimated and true value of fourier coefficients
error_exp = np.absolute(c1 - coeff_exp)
error_coscos = np.absolute(c2 - coeff_coscos)

#find the maximum error
max_error_exp = np.max(error_exp)
max_error_coscos = np.max(error_coscos)

#plotting the error
plt.plot(range(51), error_exp, 'ro', label='Error for exp(x)')
plt.title('Plot of error vs n for exp(x)')
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.show()

plt.plot(range(51), error_coscos, 'ro', label='Error for cos(cos(x))')
plt.title('Plot of error vs n for cos(cos(x))')
plt.xlabel('n')
plt.ylabel('Error')
plt.legend()
plt.grid()
plt.show()

#maximum error
print('Maximum error for exp(x) is: ', max_error_exp)
print('Maximum error for cos(cos(x)) is: ', max_error_coscos)

#Question 7

#Plotting the estimated functions A*c1 and A*c2
b1_est = np.dot(A, c1)
b2_est = np.dot(A, c2)

plt.semilogy(x, b1_est, 'go', label='Estimated function for exp(x)', alpha = 0.5)
plt.semilogy(x, b1, 'k', label='True Value')
plt.title('Plot of estimated function vs x for exp(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()

plt.plot(x, b2_est, 'go', label='Estimated function for cos(cos(x))', alpha = 0.5)
plt.plot(x, b2, 'k', label='True Value')
plt.title('Plot of estimated function vs x for cos(cos(x))')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()














