import numpy as np
from scipy import signal
from scipy.signal import square , sawtooth
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.cluster import Birch ,DBSCAN
'''
Most Recent Revision : 07/09/22
    -Added in transform/inverse transform functionality
'''



def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y


class Clustered_ICA():
  from sklearn.decomposition import PCA, FastICA
  from scipy import signal
  from sklearn.cluster import Birch ,DBSCAN
  '''
  This class performs fast ICA and sorts the returned values by explained variance.
  The returned values are scaled by default to a percentage variance explained.
  
  '''
  def __init__(self,X,num_comp = "", max_iter = 200,tol = 0.0001,scale = 100, sort = "DBSCAN",filt = True):
    from sklearn.decomposition import FastICA
    self.DBSCAN = DBSCAN
    self.Birch
    self.X = X
    self.max_iter = max_iter
    self.tol = tol
    total_comps = np.min(np.shape(X))
    self.total_comps = total_comps
    ICA = FastICA(total_comps,max_iter = self.max_iter,tol = self.tol)
    X_transformed = ICA.fit_transform(X).T
    if num_comp == "":
      self.sort = "DBSCAN"
      model = self.DBSCAN(eps=0.3, min_samples=10)
      model.fit(X_transformed)
      #Now we can use this model to predict classes
      Yhat = model.fit_predict(X_transformed)
      clusters = np.unique(Yhat)
      Points_per_Cluster = []
    else:
      self.sort = "Birch"
      model = self.Birch(threshold=0.3, n_clusters=num_comp)
      model.fit(X_transformed)
      #Now we can use this model to predict classes
      Yhat = model.fit_predict(X_transformed)
      clusters = np.unique(Yhat)
      #So Lets Actually Make a thing to Plot Things Properly
    Points_per_Cluster = []
    ICs = []
    Points_per_Cluster = []
    for cl in clusters:
      pts = np.where(Yhat == cl)[0]
      Points_per_Cluster.append([cl,len(pts)])
      IC = X_transformed[pts]
      IC = np.sum(IC,axis = 0)
      self.IC0 = IC
      if filt == True:
        filtered_IC = butter_lowpass_filter(IC,30,1500)
        IC = filtered_IC
        self.unfiltIC = self.IC0
      ICs.append(IC.T)
    self.Points_Per_Cluster = Points_per_Cluster
    PTS = []
    for i in self.Points_Per_Cluster:
      PTS.append(i[1])

    PTS = np.array(PTS)
    n = np.sum(PTS)
    self.Explained_Percentage = PTS/n*100
    self.ICs = ICs
    self.mixing_matrix = ICA.mixing_  # Get estimated mixing matrix
    self.inverse_transform = ICA.inverse_transform

  def fine_tune_filtering(self,cutoff,fs):
    self.filtered_IC = butter_lowpass_filter(self.ICs,cutoff,fs)
    return self.filtered_IC



class minimumSortedRankedICA():
  from sklearn.decomposition import PCA, FastICA
  from scipy import signal
  import warnings
  warnings.filterwarnings('ignore')
  '''
  This class performs fast ICA and sorts the returned values by explained variance.
  The returned values are scaled by default to a percentage variance explained.
  
  '''
  def __init__(self,X,tol = 10):
    '''
    First we do the ICA
    '''  
    minScore = 100
    n = 1
    while minScore > tol:
      n+=1
      ica = FastICA(n_components = n)
      XICA = ica.fit_transform(X)
      ICS = XICA.T
      '''
      Then We Rank scores
      '''
      innerProducts = []
      for IC in ICS:
        inprod = betterInnerProduct(IC,X)
        innerProducts.append(inprod)

      tot = np.sum(np.abs(innerProducts))
      innerProducts = np.abs(innerProducts)/tot
      Sorting = innerProducts.argsort()
      Sorting = Sorting[::-1]
      scores = innerProducts[Sorting]*100
      minScore = scores[-1]

    self.scores = innerProducts[Sorting]*100

    scaledICs = []

    sortedProducts = innerProducts[Sorting]
    sortedICs = ICS[Sorting]
    for i in range(len(innerProducts)):
      scaled = sortedICs[i]*sortedProducts[i]
      scaledICs.append(scaled)

    scaledICs = np.array(scaledICs)/np.max(scaledICs)
    self.ICs = scaledICs
    self.mixing_matrix = ica.mixing_  # Get estimated mixing matrix
    self.inverse_transform = ica.inverse_transform
    self.transform = ica.transform

def mean_square_error(y,yhat):
  y = np.array(y)
  yhat = np.array(yhat)
  N = len(y)
  E =N**-1* sum(abs(y-yhat))**2
  return E

  #construct here the function
def signal_deconstruction_hankel(F,L):
  '''constructs the L-Trajectory Matrix X given an input (F) and shift length (L)'''
  L = int(L)
  N = len(F)
  K = N - L +1
  X  = np.zeros([K,L])

  for i in range(K):
    j = i+1

    X[i] = F[j-1:j+L-2+1]

  return X.T

def inverse_hankel(X):
  [L,K] = np.shape(X)
  N = K+L -1
  F = np.zeros(N)
  X = np.array(X)
  X = X.T
  for i in range(K):
    j = i+1
    F[j-1:j+L-2+1] = X[i]

  return F
  
def norm(x):
  return np.sum(np.abs(x))

def innerProductAngle(x,y):
  dots = np.dot(x,y)
  norms = norm(x)*norm(y)
  return np.arccos(dots/norms)

def betterInnerProduct(x,y):
  return np.sum(np.dot(x,y))

class sortedRankedICA():
  from sklearn.decomposition import PCA, FastICA
  from scipy import signal
  import warnings
  warnings.filterwarnings('ignore')
  '''
  This class performs fast ICA and sorts the returned values by explained variance.
  The returned values are scaled by default to a percentage variance explained.
  
  '''
  def __init__(self,X,n):
    '''
    First we do the ICA
    '''  
    ica = FastICA(n_components = n)
    XICA = ica.fit_transform(X)
    ICS = XICA.T
    '''
    Then We Rank scores
    '''
    innerProducts = []
    for IC in ICS:
      inprod = betterInnerProduct(IC,X)
      innerProducts.append(inprod)

    tot = np.sum(np.abs(innerProducts))
    innerProducts = np.abs(innerProducts)/tot
    Sorting = innerProducts.argsort()
    Sorting = Sorting[::-1]

    self.scores = innerProducts[Sorting]*100

    scaledICs = []

    sortedProducts = innerProducts[Sorting]
    sortedICs = ICS[Sorting]
    for i in range(len(innerProducts)):
      scaled = sortedICs[i]*sortedProducts[i]
      scaledICs.append(scaled)

    scaledICs = np.array(scaledICs)/np.max(scaledICs)
    self.ICs = scaledICs
    self.mixing_matrix = ica.mixing_  # Get estimated mixing matrix
    self.inverse_transform = ica.inverse_transform
    self.transform = ica.transform



def prettyplot(Y = 0, x = 0,FS = [10,7], Xlabel = 'x',Ylabel = 'y',Title = 'Graph',newFig = True,graphType = "plot"):
    '''Function to plot things, takes Main data to be plotted first, ie y,x where y = f(x)
    '''
    if type(Y) ==int:
        
        return print("Please Add Data")
    if type(x)==int:
        x = range(len(Y))
    if newFig:
        plt.figure(figsize = FS)
    if graphType == "plot":    
        plt.plot(x,Y)
    if graphType == "hist":
        plt.hist(Y)
    plt.title(Title)
    plt.xlabel(Xlabel)
    plt.ylabel(Ylabel)

