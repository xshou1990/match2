"""
treatment effect metrics

@author Xiao Shou

"""

import numpy as np
from sklearn import metrics,decomposition
import numpy as np
import pandas as pd
import scipy.spatial
#from lpproj import LocalityPreservingProjection
from sklearn.random_projection import GaussianRandomProjection
from scipy import linalg as LA
from sklearn.covariance import EmpiricalCovariance
from scipy.linalg import sqrtm



def PSM_ATT(data,propensity):
    '''
    implementation of Propensity scores by L1-regularized Logistic Regression and its ATT 
    '''
    
    treatment_pred = propensity[data['assignment']==1]
    control_pred = propensity[data['assignment']==0]
    treatment_outcome = data[data['assignment']==1]['outcome'].values
    control_outcome = data[data['assignment']==0]['outcome'].values
    dist_NNM = scipy.spatial.distance.cdist(treatment_pred, control_pred,metric='euclidean')
    NNM = np.argmin(dist_NNM,axis=1)
    control_matched = control_outcome[NNM]  
    
    return np.mean(treatment_outcome - control_matched)

def Eucl_ATT(data):
    '''
    implementation of Matching via Euclidean Distance in original space
    '''
    X = data.iloc[:,:-2].values
    transD1 = X[data['assignment']==1]
    transD0 = X[data['assignment']==0]
    dist_NNM = scipy.spatial.distance.cdist(transD1, transD0,metric='euclidean')
    NNM = np.argmin(dist_NNM,axis=1)
    ATT = np.mean(data[data['assignment']==1]['outcome'].values - data[data['assignment']==0]['outcome'].values [NNM] )
    return ATT

def Maha_ATT(data):
    '''
    implementation of Matching via Mahalanobis Distance
    '''
    X = data.iloc[:,:-2].values
    cov = EmpiricalCovariance().fit(X[data['assignment']==0])
    precision_mat = cov.get_precision()
    G = sqrtm(precision_mat)
    transD1 = np.matmul(G,(X[data['assignment']==1]).T)
    transD0 = np.matmul(G,(X[data['assignment']==0]).T)
    dist_NNM = scipy.spatial.distance.cdist(transD1.T, transD0.T,metric='euclidean')
    NNM = np.argmin(dist_NNM,axis=1)
    ATT = np.mean(data[data['assignment']==1]['outcome'].values - data[data['assignment']==0]['outcome'].values [NNM] )
    return ATT


def RNNM_ATT(data, components):
    '''
    implementation of Matching via Dimensionality Reduction for Estimation of Treatment Effects in Digital Marketing Campaigns
    Sheng Li et al (IJCAI-16)
    '''
    ATT = []
    for i in range(50):
        transformer = GaussianRandomProjection(n_components=components)
        X = transformer.fit_transform(data.iloc[:,:-2])

        dim_name = []
        for i in range(components):
            dim_name.append('dim{}'.format(i+1))        
        X = pd.DataFrame(X, columns = dim_name)  
        data1 = pd.concat([data, X], axis=1)

        xty_treatment = data1[data1['assignment']==1][['outcome']+ dim_name]
        xty_control = data1[data1['assignment']==0][['outcome']+ dim_name]

        dist_NNM = scipy.spatial.distance.cdist(xty_treatment[dim_name], xty_control[dim_name],metric='euclidean')
        NNM = np.argmin(dist_NNM,axis=1)
        #print(dist_NNM)
        xty_control_matched = xty_control.iloc[NNM,:]    
        ATT.append(np.mean(xty_treatment['outcome'].values - xty_control_matched['outcome'].values))
        
    
    return np.median(ATT)

def LPP_ATT(data, components):
    '''
    Implementation of Locality Preserving Projection and its ATT
    '''

    lpp = LocalityPreservingProjection(n_components=components)
    X = lpp.fit_transform(data.iloc[:,:-2])
    
    dim_name = []
    for i in range(components):
        dim_name.append('dim{}'.format(i+1))        
    X = pd.DataFrame(X, columns = dim_name)  
    data1 = pd.concat([data, X], axis=1)
    
    xty_treatment = data1[data1['assignment']==1][['outcome']+ dim_name]
    xty_control = data1[data1['assignment']==0][['outcome']+ dim_name]
    
    dist_NNM = scipy.spatial.distance.cdist(xty_treatment[dim_name], xty_control[dim_name],metric='euclidean')
    NNM = np.argmin(dist_NNM,axis=1)
    #print(dist_NNM)
    xty_control_matched = xty_control.iloc[NNM,:]    
    ATT = np.mean(xty_treatment['outcome'].values - xty_control_matched['outcome'].values)   
    
    return ATT

def PCA_ATT(data, components):
    '''
    Implementation of PCA and its ATT
    '''
    pca = decomposition.PCA(n_components=components)
    pca.fit(data.iloc[:,:-2])
    X = pca.transform(data.iloc[:,:-2])
    
    dim_name = []
    for i in range(components):
        dim_name.append('dim{}'.format(i+1))        
    X = pd.DataFrame(X, columns = dim_name)  
    data1 = pd.concat([data, X], axis=1)
    
    xty_treatment = data1[data1['assignment']==1][['outcome']+ dim_name]
    xty_control = data1[data1['assignment']==0][['outcome']+ dim_name]
    
    dist_NNM = scipy.spatial.distance.cdist(xty_treatment[dim_name], xty_control[dim_name],metric='euclidean')
    NNM = np.argmin(dist_NNM,axis=1)
    #print(dist_NNM)
    xty_control_matched = xty_control.iloc[NNM,:]    
    ATT = np.mean(xty_treatment['outcome'].values - xty_control_matched['outcome'].values)   
    
    return ATT


def aveNN_ATT(data,position,dist_metric):
    ''' 
    Implementation of Match^2's Self Organizing Map and its ATT 
    '''
    unzipped_pos = np.array(list(zip(*position)))
    dim_name = []
    for i in range(unzipped_pos.shape[0]):
        dim_name.append('dim{}'.format(i+1))
    unzipped_pos = pd.DataFrame(unzipped_pos.T, columns = dim_name)    
    
    data1 = pd.concat([data, unzipped_pos], axis=1)
    #print(data1)
    xty_treatment = data1[data1['assignment']==1.0][['outcome']+ dim_name]
    xty_control = data1[data1['assignment']==0.0][['outcome']+ dim_name]
    
    xty_treatment.reset_index(drop = True, inplace = True)
    
    #print(xty_treatment.shape)
    dist_NNM = scipy.spatial.distance.cdist(xty_treatment[dim_name], xty_control[dim_name],metric=dist_metric)
    ATT = []
    for i in range(dist_NNM.shape[0]):
        NNM = np.argwhere(dist_NNM[i] == np.amin(dist_NNM[i]))
        #print(len(NNM))
        ATT.append(xty_treatment['outcome'][i]- np.mean(xty_control.iloc[np.squeeze(NNM,axis=1),:]['outcome']))
        #print(ATT[i])
    
    return np.mean(ATT)


def Tau_G(data,position,dist_metric):
    ''' 
    Implementation of Match^2's Self Organizing Map and its global matching quality score Tau_G
    '''

    unzipped_pos = np.array(list(zip(*position)))
    dim_name = []
    for i in range(unzipped_pos.shape[0]):
        dim_name.append('dim{}'.format(i+1))
    unzipped_pos = pd.DataFrame(unzipped_pos.T, columns = dim_name)    

    data1 = pd.concat([data, unzipped_pos], axis=1)
    #print(data1)
    xty_treatment = data1[data1['assignment']==1.0][['outcome']+ dim_name]
    xty_control = data1[data1['assignment']==0.0][['outcome']+ dim_name]

    xty_treatment.reset_index(drop = True, inplace = True)

    #print(xty_treatment.shape)
    dist_NNM = scipy.spatial.distance.cdist(xty_treatment[dim_name], xty_control[dim_name],metric=dist_metric)

    tau_g = 0
    for i in range(dist_NNM.shape[0]):
        tau_i = len(np.where(dist_NNM[i] == dist_NNM[i].min())[0])/(dist_NNM[i].min()+1)
        tau_g += tau_i

    return tau_g/dist_NNM.shape[0]



def aveNN_ATU(data,position,dist_metric):
    ''' 
    Implementation of Match^2's Self Organizing Map and its ATU 
    '''
    unzipped_pos = np.array(list(zip(*position)))
    dim_name = []
    for i in range(unzipped_pos.shape[0]):
        dim_name.append('dim{}'.format(i+1))
    unzipped_pos = pd.DataFrame(unzipped_pos.T, columns = dim_name)    
    
    data1 = pd.concat([data, unzipped_pos], axis=1)
    #print(data1)
    xty_treatment = data1[data1['assignment']==1.0][['outcome']+ dim_name]
    xty_control = data1[data1['assignment']==0.0][['outcome']+ dim_name]
    
    xty_control.reset_index(drop = True, inplace = True)
    
    #print(xty_treatment.shape)
    dist_NNM = scipy.spatial.distance.cdist( xty_control[dim_name], xty_treatment[dim_name], metric=dist_metric)
    ATU = []
    for i in range(dist_NNM.shape[0]):
        # NNM: index of control being matching with treated
        NNM = np.argwhere(dist_NNM[i] == np.amin(dist_NNM[i]))
        #print(len(NNM))
        ATU.append( np.mean(xty_treatment.iloc[np.squeeze(NNM,axis=1),:]['outcome']) - xty_control['outcome'][i])
        #print(ATT[i])
    
    return np.mean(ATU)


    
def BNR_ATT2(xty,alpha,beta,dim): ### incorrect implementation ( eqn 5 from paper incorrect)
    '''
    implementation of Balanced Nonlinear Representations with two classes
    Li, Sheng, and Yun Fu. "Matching on balanced nonlinear representations for treatment effects estimation." 
    In Advances in Neural Information Processing Systems, pp. 929-939. 2017.
    '''

    bins = 2
    group_names = [1,2]
    xty['outcome_bin'] = pd.qcut(xty['outcome'], bins, labels=group_names)

    data = xty.iloc[:,:-3].values

    class1 = data[xty['outcome_bin']==1]
    class2 = data[xty['outcome_bin']==2]
   # class3 = data[xty['outcome_bin']==3]
    Xc = data[xty['assignment']==0]
    Xt = data[xty['assignment']==1]

    Yc = xty['outcome'][xty['assignment']==0].reset_index(drop = True)
    Yt = xty['outcome'][xty['assignment']==1].reset_index(drop = True)

    ksiall = scipy.spatial.distance.cdist(data, data, metric='euclidean')
    ksi1 = scipy.spatial.distance.cdist(data, class1, metric='euclidean')
    ksi2 = scipy.spatial.distance.cdist(data, class2, metric='euclidean')
    #ksi3 = scipy.spatial.distance.cdist(data, class3, metric='euclidean')

    kc = scipy.spatial.distance.cdist(data, Xc, metric='euclidean')
    kt = scipy.spatial.distance.cdist(data, Xt, metric='euclidean')

    distcc = scipy.spatial.distance.cdist(Xc, Xc, metric='euclidean')
    distct = scipy.spatial.distance.cdist(Xc, Xt, metric='euclidean')
    disttt = scipy.spatial.distance.cdist(Xt, Xt, metric='euclidean')

    kall = np.exp(-0.02*ksiall**2) 
    k1 = np.exp(-0.02*ksi1**2) 
    k2 = np.exp(-0.02*ksi2**2) 
    #k3 = np.exp(-0.02*ksi3**2) 
    Kc = np.exp(-0.02*kc**2)  
    Kt = np.exp(-0.02*kt**2)  

    kcc = np.exp(-0.02*distcc**2)
    kct = np.exp(-0.02*distct**2)
    ktt = np.exp(-0.02*disttt**2)
    ktc = kct.T

    K = np.block([[kcc,kct],[ktc,ktt]])

    m1 = np.mean(k1,axis=1)
    m2 = np.mean(k2,axis=1)
   # m3 = np.mean(k3,axis=1)
    mbar = np.mean(kall,axis=1)

    KI =  np.exp(1)*np.matmul(np.expand_dims((m1-m2),axis=1),np.expand_dims((m1-m2),axis=1).T)
    #+ \
    #np.exp(2)*np.matmul(np.expand_dims((m1-m3),axis=1),np.expand_dims((m1-m3),axis=1).T) + \
    #np.exp(1)*np.matmul(np.expand_dims((m2-m3),axis=1),np.expand_dims((m2-m3),axis=1).T)

    KW = 1.0/data.shape[0] * (np.matmul((k1-np.expand_dims(mbar,axis=1)),(k1-np.expand_dims(m1,axis=1)).T) + \
    np.matmul((k2-np.expand_dims(mbar,axis=1)),(k2-np.expand_dims(m2,axis=1)).T)) #+ \
   # np.matmul((k3-np.expand_dims(mbar,axis=1)),(k3-np.expand_dims(m3,axis=1)).T))

    Lcc = 1/Xc.shape[0]**2 * np.ones((kcc.shape[0],kcc.shape[1]))
    Ltt = 1/Xt.shape[0]**2 * np.ones((ktt.shape[0],ktt.shape[1]))
    Lct = -1/(Xt.shape[0]*Xc.shape[0]) * np.ones((kct.shape[0],kct.shape[1])) 
    Ltc = Lct.T
    L = np.block([[Lcc,Lct],[Ltc,Ltt]])

    sum_mat = KI - alpha*KW - beta*np.matmul(np.matmul(K,L),K)
    _, P = LA.eigh(sum_mat, eigvals = [sum_mat.shape[0]-dim,sum_mat.shape[0]-1])
    Xc_hat = np.matmul(P.T,Kc)
    Xt_hat = np.matmul(P.T,Kt)
    dist_NNM = scipy.spatial.distance.cdist(Xt_hat.T, Xc_hat.T, metric='euclidean')
    NNM = np.argmin(dist_NNM,axis=1)

    return np.mean(Yt - Yc[NNM])


