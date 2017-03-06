#make a class of SORN
# Intrinsic Plasticity
from numpy import *
import numpy as np
from sklearn.preprocessing import normalize
import sys
import os    #os._exit(0) for quitting ipdb
import ipdb
from scipy import sparse
from scipy.sparse import csc_matrix
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA


def binary(x, Threshold):
    x[x >= Threshold] = 1
    x[x < Threshold] = 0
    return x

class SORN(object):
    def __init__(self, N_e, ifsparse=False):
        self.N_e = np.int(N_e)   #number of excitatory neurons
        self.N_u = np.int(0.05 * N_e)
        self.N_i = np.int(0.2 * N_e)  #number of inhitatory neurons
        self.T_e_min = 0.
        self.T_e_max = 0.5
        self.T_e = np.random.rand(self.N_e) * self.T_e_max
        self.T_i_min = 0.8
        self.T_i_max = 1.0
        self.T_i = np.random.rand(self.N_i) * self.T_i_max
        self.eta_IP = 0.001
        self.eta_STDP = 0.001
        self.H_IP = 2 * self.N_u / self.N_e  #target activity
        self.Lambda = 10            #average there are 10 connections per neuron
        self.lam = 1 - float(self.Lambda)/self.N_e
        self.ifsparse = ifsparse
        # Initialize weight matrices
        # W_to_from (W_ie = from excitatory to inhibitory)
        nn = 1
        while nn > 0:
            self.W_ee = np.random.rand(self.N_e, self.N_e)
            self.W_ee[self.W_ee < self.lam] = 0
            self.W_ee[self.W_ee > 0] = np.random.rand(sum(self.W_ee > 0))     #sum(W_ee > 0) get how many elements of W_ee is >0
            summe = np.sum(self.W_ee, axis = 1)
            nn = sum(summe == 0)
            if nn > 0:
                break
        self.W_ee = self.W_ee * (np.ones((self.N_e, self.N_e)) - np.eye(self.N_e))   #make the diag to be 0
        self.W_ee = normalize(self.W_ee, norm="l1", axis = 1 , copy=False)

        if self.ifsparse:
            self.W_ee = sparse.csc_matrix(self.W_ee)
            self.row = self.W_ee.indices
            self.col = np.repeat(arange(self.N_e),np.diff(self.W_ee.indptr))
            self.data = self.W_ee.data
        else:
            self.W_ee = self.W_ee
        self.W_ei = np.random.rand(self.N_e, self.N_i)         #weight from excitatory to inhitatory neurons
        self.W_ie = np.random.rand(self.N_i, self.N_e)        #weight from inhitatory to excitatory neurons
        self.W_ei = normalize(self.W_ei , norm='l1' ,  axis = 1 , copy=False)
        self.W_ie = normalize(self.W_ie , norm='l1' ,  axis = 1 , copy=False)

    def run(self, steps, W_ee, U_actv,  toReturn=[], training = False, testing = False):
        """
        Simulates SORN for a defined number of steps

        Parameters:
            N: int
                Simulation steps
           U_actv: int
                input activation of neuron clusters
            toReturn: list
                Tracking variables to return. Options are: 'W_ee','stateX'
        return:
            ans: a dictionary of variables that you put in toReturn list
        """
        ans = {}   #create a diction of return values
        x_pre = np.random.rand(self.N_e)
        x_pre = binary(x_pre, self.T_e_max)
        stateX = np.random.rand(self.N_e, steps+1)
        stateX[:, 0] = x_pre
        y_pre = np.random.rand(self.N_i)
        y_pre = binary(y_pre, self.T_i_max)
        for t in range(steps-1):
            Rx = self.W_ee.dot(stateX[:,t]) - np.dot(self.W_ei, y_pre) - self.T_e +  U_actv[:,t]
            Rx = binary(Rx, 0)
            stateX[:,t + 1] = Rx
            y_t = np.dot(self.W_ie, Rx) - self.T_i
            y_t = binary(y_t, 0)
            y_pre = y_t
            if not training or testing:
                #STDP
                if self.ifsparse:
                    self.data += self.eta_STDP * (stateX[self.row,t + 1] * stateX[self.col,t] - stateX[self.row,t] * stateX[self.col,t + 1])
                    self.data[self.data < 0] = 0
                    # SN:
                    row_sums = np.array(self.W_ee.sum(1))[:,0]
                    self.data /= row_sums[self.row]
                else:
                    A = stateX[:,t].dot(stateX[:,t + 1].T)
                    self.W_ee += self.eta_STDP * (A.T - A)
                    self.W_ee[self.W_ee < 0] = 0

                    #SN:
                    self.W_ee = normalize(self.W_ee , norm='l1' ,  axis = 1 , copy=False)

            # IP:
            self.T_e = self.T_e + self.eta_IP * (stateX[:,t] - self.H_IP)

        if 'W_ee' in toReturn:
                ans['W_ee']= self.W_ee
        if 'stateX' in toReturn:
                ans['stateX']= stateX
        return ans

    def majorityVote(self, segment_cue, bi_cal_out_matrix, desire_out_matrix, label_digit):
        """
        Get performance by comparing the calculated output with the desired output and get a vote in a period of time(provided by segment_cue)

        Parameters:
            segment_cue: 2D array (2*steps)
                first row is for reference, is the order of digits presenting to the network
                second row is how many frames of each digit
            cal_out: 2D array  (digits*(steps+1))
                output calculated with W_out
            desire_out_matrix: 2D array  (digits*steps)
                Tracking variables to return. Options are: 'W_ee','stateX'
        return:
            ans: a dictionary of variables that you put in toReturn list
        """
        count = 0
        segment_cue = segment_cue
        label_digit = label_digit
        segment_cue[0] = segment_cue[0] - 1     #using prior knowledge get the index of changing digit
        bi_cal_label_decode = np.nonzero(bi_cal_out_matrix.T==1)[1]    #the same way to get the digit
        seg_start = 0
        cal_majority_takeover = np.array([])
        for length in segment_cue:
            bias = 2
            cal_votes = np.bincount(bi_cal_label_decode[seg_start+bias: seg_start+length-2*bias])  # get the sum only in length
            cal_majority_takeover= np.append(cal_majority_takeover, np.argmax(cal_votes))
            seg_start = seg_start + length
        count = sum(cal_majority_takeover==label_digit) #sum column-wise how many trues are there
        performance= count* 1.0 / (cal_majority_takeover.shape[0])
        return performance, cal_majority_takeover

    def plot_confusion_matrix(self, cm, classes, title='Confusion matrix', cmap='gray_r', normalize=False):#plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.figure()
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def visualizeMatrices(self, kwargs):
        '''
        input: dic of key and value of what you want to visualize
        '''
        for key, value in kwargs.iteritems():
            if key == 'W_EE':
                plt.figure()
                plt.imshow(kwargs[key].todense(), interpolation='nearest', cmap='Greys')
                plt.spy(kwargs[key], precision=0.1, markersize=5)
                plt.title('W_ee N_e=%d' % self.N_e)
                plt.colorbar()
            elif key == 'stateX':
                plt.figure()
                plt.imshow(kwargs[key][:, 0:1000], interpolation='nearest', cmap='GnBu')
                plt.colorbar()
            elif key == 'W_OUT':
                plt.figure()
                plt.imshow(kwargs[key][:,0:100], interpolation='nearest', cmap='Blues')
                plt.title('W_out N_e=%d' % self.N_e)
                plt.colorbar()
            else:
                print ("Nothing to print")

    def visualPCA(self, stateX, label, steps, n_conponents=10):
        pcaX = stateX.T
        pcaY = label[0:steps]
        fig = plt.figure()
        ax = Axes3D(fig, rect=[0, 0, 0.95, 1], elev=48, azim=134)
        plt.cla()
        pca = PCA(n_conponents)
        pca.fit(pcaX)
        pcaX = pca.transform(pcaX)

        for label in np.arange(n_conponents):
            ax.text3D(pcaX[pcaY == label, 0].mean(),
                      pcaX[pcaY == label, 1].mean() + 1.5,
                      pcaX[pcaY == label, 2].mean(), label,
                      horizontalalignment='center',
                      bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
        # Reorder the labels to have colors matching the cluster results
        #pcaY = np.choose(pcaY.astype(int), np.arange(n_conponents)+5).astype(np.float)
        ax.scatter(pcaX[0:steps*0.6, 0], pcaX[0:steps*0.6, 1], pcaX[0:steps*0.6, 2],  cmap='cool' )

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_title("PCA, NG={}, alpha={}".format(10, 80))
        

