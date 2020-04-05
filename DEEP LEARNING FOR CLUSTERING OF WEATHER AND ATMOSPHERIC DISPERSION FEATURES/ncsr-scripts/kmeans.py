# Run: kmeans.py <GHT_ALL.npz> <LEVEL> <PREFIX> ||python2 kmeans.py ../GHT_all.npz 700 /workspace/products/raw_kmeans/
import sys
import numpy as np
import os
sys.path.append('..')
from vis import plot_tsne
from sklearn.manifold import TSNE

import utils
from tsne import closest

from Clustering import Clustering
from Dataset_transformations import Dataset_transformations
import dataset_utils 

from netcdf_subset import netCDF_subset


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


# CONST.
GHT_FILE = sys.argv[1]
LEVELS_STR = ['500', '700', '900']
LEVELS = ['G5', 'G7', 'G9']
LEVEL = sys.argv[2]

PREFIX = sys.argv[3]

CONFIG_NAME = 'raw_kmeans'

_LEVEL = LEVELS[LEVELS_STR.index(LEVEL)]

# Load GHT level
data = np.load(GHT_FILE)[_LEVEL]

# Make 2 days (overlap 1 day)
# data, _, _ = utils.overlap(data, win = 4, t = 8)



# Reshape
# data = data.reshape(data.shape[1] * data.shape[2], data.shape[0])
ds = Dataset_transformations(data, 1000, data.shape)
if os.path.exists(PREFIX+CONFIG_NAME+'.zip'):
    clust_obj = dataset_utils.load_single(PREFIX+CONFIG_NAME+'.zip')
else:
    print 'Doing kmeans.....'
    clust_obj = Clustering(ds,n_clusters=15,n_init=100,features_first=False)
    clust_obj.batch_kmeans(10)
    print 'Saving .....'
    clust_obj.save(PREFIX+CONFIG_NAME+'.zip')

print(data.shape)
z = TSNE(n_components=2, verbose=2).fit_transform(data)
pred_kmeans = clust_obj._link.predict(data)
plot_tsne(z,pred_kmeans,data)
exit()
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
