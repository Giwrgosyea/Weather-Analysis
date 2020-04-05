import numpy as np
from transforms3d.axangles import axangle2mat
from math import sqrt
from numpy import array
from scipy.interpolate import CubicSpline
from scipy.interpolate import interp1d
from random import sample 
import random 

##DFSDFSDFSDFdadasdasdasasdda
def zero_interpolation(X,rate):
    Xx=[i for i in range(0,len(X),1)]
    zero_sample=sample(Xx,int(len(Xx)*rate))
    print("size of zeros..",len(zero_sample))
    
    for i in range(len(X)):
        if i in zero_sample:
            X[i]=0
    np.save("zero_interpolation.npy",X)

    print("zero_interpolation",X.mean(),X.shape)
    return X

def random_interpolation(X,step):
    count=0
    x=list() 
    while count < X.shape[0]: 
        print("count",count)
        x.append(X[count])
        rand=random.choice([0,2,3,4]) 
        j=(count+1)+rand 
        if rand!=0 and j < X.shape[0]: 
            inter=interp1d([X[count].min(),X[j].max()],np.vstack([X[count],X[j]]),axis=0,fill_value="extrapolate") ##start point /end point 
            for k in range(rand):
                #print('---->',k)
                x.append(inter(random.uniform(1,1.2)))
             
            for i in range(1,step,1):
                if len(x) < X.shape[0]:
                    x.append(X[count+i])
            count=j+step-1
        else:
            count+=1 
    print("random_interpolation",np.array(x).mean(),np.array(x).shape)
    np.save("random_interpolation.npy",np.array(x))
    return  np.array(x)  


def interpolation(X):
    x=list()
    for i in range(0,len(X),2):
        j= i+2
        #rint("--->",i)

        if j<len(X):
            x.append(list(X[i]))
            inter=interp1d([X[i].min(),X[j].max()],np.vstack([X[i],X[j]]),axis=0,fill_value="extrapolate") ##start point /end point

            x.append(inter(random.uniform(1,1.2)))
            #x.append(list(X[j]))
        else:
            x.append(list(X[i]))

    x=np.array(x)
    print("data after interpolation:..",x.shape,x.mean())
    np.save("interpolation.npy", x)
    return x
        

def gauss_noise(X,clip="false"):
    # Generate corrupted data by adding noise with normal dist
    # centered at 0.5 and std=0.5
    if X.mean() <0:

        loc=-1*X.mean() * 100
        scale=-1*X.mean() * 100
    else:

        loc=X.mean() * 0.1
        scale=X.mean() * 0.1
    noise = np.random.normal(loc=loc, scale=scale, size=X.shape)
    X_noise = X + noise
    if clip=="true":
        x_train_noisy = np.clip(x_train_noisy, 0., 1.)
    np.save("X_noisy",np.array(X_noise))
    return X_noise

def DA_Jitter(X, sigma=0.05):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    np.save("X_jitter",np.array(X+myNoise))
    print("DA_Jitter",np.array(X+myNoise).shape,np.array(X+myNoise).mean)

    return X+myNoise

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    np.save("X_scaled",np.array(X*myNoise))
    print("X_scaled",np.array(X*myNoise).shape,np.array(X*myNoise).mean)
    return X*myNoise


def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X,axangle2mat(axis,angle))

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_x = CubicSpline(xx[:,0], yy[:,0])
    cs_y = CubicSpline(xx[:,1], yy[:,1])
    cs_z = CubicSpline(xx[:,2], yy[:,2])
    return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()

 