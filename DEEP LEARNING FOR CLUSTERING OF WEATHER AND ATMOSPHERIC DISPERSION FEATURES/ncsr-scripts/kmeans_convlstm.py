# Run: kmeans.py <GHT_ALL.npz> <PREFIX> <model>  ||python2 kmeans.py ../GHT_all.npz  /workspace/products/raw_kmeans/ model 
# K-means for Convolution LSTM model
import sys

import os
import numpy as np
sys.path.append('..')
import utils
from tsne import closest

from Clustering import Clustering
from Dataset_transformations import Dataset_transformations
import dataset_utils 

from netcdf_subset import netCDF_subset
import demo


def desc_date(clust_obj,nc_subset,time_idx):
    desc_date = []
    for pos,i in enumerate(time_idx):
        gvalue = nc_subset._dataset.variables[nc_subset._time_name][i[0][0]]
        sim_date = ""
        for gv in gvalue:
            sim_date += gv
        desc_date.append(sim_date)
    clust_obj._desc_date = desc_date

NC_PATH = '/workspace/wrfhy/wrfvol/40years.nc'

TIMESTEPS = 8

# CONST.
GHT_FILE = sys.argv[1]
LEVELS_STR = ['500', '700', '900']
LEVELS = ['G5', 'G7', 'G9']
# LEVEL = sys.argv[2]

PREFIX = sys.argv[2]

MODEL = sys.argv[3]


CONFIG_NAME = 'ght_convlstm_'


# Load GHT level
# ght_500 = np.load(GHT_FILE)[LEVELS[LEVELS_STR.index('500')]]
# ght_500=ght_500.reshape(ght_500.shape[0],1,ght_500.shape[1])

ght_700 = np.load(GHT_FILE)[LEVELS[LEVELS_STR.index('700')]]
# ght_700=ght_700.reshape(ght_700.shape[0],1,ght_700.shape[1])

# ght_900 = np.load(GHT_FILE)[LEVELS[LEVELS_STR.index('900')]]
# ght_900=ght_900.reshape(ght_900.shape[0],1,ght_900.shape[1])

# print('Info fot ght 500...',ght_500.shape)
print('Info fot ght 700...',ght_700.shape)
# print('Info fot ght 900...',ght_900.shape)


# data=np.concatenate((ght_500,ght_700),axis=1)
# data=np.concatenate((data,ght_900),axis=1)
# print(data.shape)

# data=data.reshape(ght_700.shape[0],data.shape[1]*data.shape[2])

# print('Info Data...',data.shape,data.mean())

seq_in,_,_=demo.overlap(ght_700,4,t=8)
seq_in = seq_in.reshape((seq_in.shape[0], seq_in.shape[1], 64, 64, 1))

if MODEL=='':
    print('insert model..')
    exit()



##predict data...
model=demo.model_conv_lstm(TIMESTEPS)
model.load_weights(MODEL)
print(model.summary())

from tensorflow.python.keras.models import Model

model = Model(inputs=model.input, outputs=model.get_layer('encoder').output)

data = model.predict(seq_in)
print("Prediction shape:",data.shape)
exit()

data = data.reshape((data.shape[0]*data.shape[1], 8*8*2))
print('reshape encoding..:',data.shape)

# Reshape
data = data.reshape(data.shape[1], data.shape[0])
ds = Dataset_transformations(data.T, 1000, data.shape)
if os.path.exists(PREFIX+CONFIG_NAME+'.zip'):
    clust_obj = dataset_utils.load_single(PREFIX+CONFIG_NAME+'.zip')
else:
    print 'Doing kmeans.....'
    clust_obj = Clustering(ds,n_clusters=15,n_init=100,features_first=False)
    clust_obj.batch_kmeans(10)
    print 'Saving .....'
    clust_obj.save(PREFIX+CONFIG_NAME+'.zip')

# Descriptor num_min: 1
num_min = 1
times_pos = closest(clust_obj._link, ds._items, num_min, win=4, t=8, save=False)
np.save(PREFIX+'time_pos_desc'+str(num_min)+'.npy', times_pos)

ns = netCDF_subset(NC_PATH, [500, 700, 900], ['GHT'], timename='Times')
desc_date(clust_obj, ns, times_pos)
clust_obj.save(PREFIX+CONFIG_NAME+'_'+str(num_min)+'.zip')

for c, i in enumerate(times_pos):
    if not os.path.exists(PREFIX+'descriptors1/'):
        os.mkdir(PREFIX+'descriptors1/')
    name = PREFIX+'descriptors1/desc_'+str(c)+'.nc'
    ns.exact_copy_file(name, i[0])

# Descriptor num_min: 10
num_min = 10
times_pos = closest(clust_obj._link, ds._items, num_min, win=4, t=8, save=False)
np.save(PREFIX+'time_pos_desc'+str(num_min)+'.npy', times_pos)

ns = netCDF_subset(NC_PATH, [500, 700, 900], ['GHT'], timename='Times')
desc_date(clust_obj, ns, times_pos)
clust_obj.save(PREFIX+CONFIG_NAME+'_'+str(num_min)+'.zip')

for c, i in enumerate(times_pos):
    frames = []
    for t in range(len(i[0])):
        i = np.array(i)
        _idc = i[:, t]
        frames.append(np.array(_idc))
    if not os.path.exists(PREFIX+'descriptors2/'):
        os.mkdir(PREFIX+'descriptors2/')
    name = PREFIX+'descriptors2/desc_'+str(c)+'.nc'
    ns.exact_copy_kmeans(name, frames)
