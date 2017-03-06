'''Force learning
f(t) = sin(x)
init network
RLS
update W
'''

#spoken digits classification with sorn
#from numpy import *
import numpy as np
import matplotlib.pyplot as plt
from pylab import *
from scipy import stats

from scipy import sparse
#from scipy.sparse import csc_matrix
from scipy.sparse import random
from scipy import signal
from class_SORN_FSD import SORN


#def activ(inputvalue, networksize):
    
    
'''simplify: WX=Y i.e. the weight matrix is always on the left'''
#init network
steps = 500
t = np.linspace(0, 10, steps)
#inputsig = signal.square(2 * np.pi * 0.5 * t)
inputsig = signal.sawtooth(2 * np.pi * 0.5 * t, width = 0.5)
#inputsig = activ(inputsig, NG)
NG = 100
sorn = SORN(NG)

# get activation
p_spars = 0.1
alpha = 80   #learning rate
w_tpre = np.random.normal(loc = 0, scale = 1/np.sqrt(NG), size=NG)
g = 1
gz = 1
np.random.seed(5)
#rvs = stats.norm(loc=0, scale = 1/np.sqrt(NG*p_spars)).rvs     #sparse init non-zero weight matrix
#W_NG = random(NG, NG, p_spars, data_rvs=rvs)

W_NG = np.random.normal(loc = 0, scale = 0.1, size=(NG, NG))
r_t = np.tanh(np.random.normal(loc = 0, scale = 0.1, size=NG))
z = np.dot(w_tpre, r_t)
P_tpre = np.eye(NG) * 1 / alpha

Rate = np.zeros((NG, steps))
Z = np.zeros((steps,))
weight = np.zeros((NG, steps))
visualize_value = ['weight', 'error', 'activity']

for i in range(steps):
    #network update according to the diffifential equation
    r_t = np.tanh(g*np.dot(W_NG, r_t) + gz*z)       #firing rate
    #updata weights min error
    print 'iteration', i

    e_tpre = np.dot(w_tpre, r_t) - inputsig[i] 
    #update weight RLS
    P_t = P_tpre - P_tpre*r_t*r_t.T*P_tpre/(1+r_t.T*P_tpre*r_t)
    w_t = w_tpre - np.dot(e_tpre * P_t, r_t)
    e_tpst = np.dot(w_t, r_t) - inputsig[i]
    
    #pass on the values for the next step
    P_tpre = P_t
    w_tpre = w_t
    
    #save the values in the matrix for visualization
    Rate[:, i] = r_t
    z = np.dot(w_t, r_t)
    Z[i] = z
    print z
    weight[:, i] = w_t
 
    #visualize weight, error, chosen neurons' activities, 
        
#PCA
sorn.visualPCA(Rate, inputsig, steps)

plt.figure()
plt.imshow(Rate, interpolation='nearest', cmap='GnBu')
plt.colorbar()
plt.title("firing rate, NG={}, alpha={}, g={}, gz={}".format(NG, alpha, g, gz))

plt.figure()
plt.plot(Rate[20:30, :].T)
plt.title("firing rate 20:30, NG={}, alpha={}, g={}, gz={}".format(NG, alpha, g, gz))

plt.figure()
plt.plot(t, Z, t, inputsig)
plt.title("Output-training, NG={}, alpha={}, g={}, gz={}".format(NG, alpha, g, gz))


plt.figure()
plt.plot(np.linalg.norm(weight[:, 1:]-weight[:, 0:-1], axis=1))
plt.title("change of weight, NG={}, alpha={}, g={}".format(NG, alpha, g))

plt.show()





