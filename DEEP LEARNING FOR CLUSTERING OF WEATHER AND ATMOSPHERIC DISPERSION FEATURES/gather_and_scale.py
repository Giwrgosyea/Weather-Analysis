# Run as:

import numpy as np
import os
import sys
from Dataset_transformations import Dataset_transformations

# npz directory
npz_dir = sys.argv[1]

# Remove "test" years
files = [i for i in os.listdir(npz_dir) if i != '1994.npz' and i != '1995.npz']

# Check files
#  for i in sorted(files):
#  print i

GHT = []
UU = []
VV = []

# Iterate trough files (GATHER)
for f in files:
    year = np.load(npz_dir + f)
    # U
    #  uu = year['UU']
    #  uu = uu[:, :, :, 0:64]
    #  # V
    #  vv = year['VV']
    #  vv = vv[:, :, 0:64, :]
    # GHT
    ght = year['GHT']
    # Gather
    GHT.append(ght)
    #  UU.append(uu)
    #  VV.append(vv)

#  Non scaled
GHT = np.array(GHT)
GHT = np.concatenate(GHT, axis=0)
print GHT.shape

#  UU = np.array(UU)
#  UU = np.concatenate(UU, axis=0)
#  print UU.shape
#
#  VV = np.array(VV)
#  VV = np.concatenate(VV, axis=0)
#  print VV.shape

_GHT = []
#  _UU = []
#  _VV = []

for i in range(GHT.shape[1]):
    # Slice level
    ght = GHT[:, i, :]
    # Flatten features
    ght = ght.reshape(ght.shape[1] * ght.shape[2], ght.shape[0])
    # Scale mean:0 std:1
    ght_ds = Dataset_transformations(ght, 1, ght.shape)
    ght_ds.normalize()
    ght = ght_ds._items.T
    # Append
    _GHT.append(ght)
print ('GHT done')

#  for i in range(UU.shape[1]):
#  # Slice level
#  uu = UU[:, i, :]
#  # Flatten features
#  uu = uu.reshape(uu.shape[1] * uu.shape[2], uu.shape[0])
#  # Scale mean:0 std:1
#  uu_ds = Dataset_transformations(uu, 1, uu.shape)
#  uu_ds.normalize()
#  uu = uu_ds._items.T
#  # Append
#  _UU.append(_UU)
#  print ('UU done')

#  for i in range(VV.shape[1]):
#  # Slice level
#  vv = VV[:, i, :]
#  # Flatten features
#  vv = uu.reshape(vv.shape[1] * vv.shape[2], vv.shape[0])
#  # Scale mean:0 std:1
#  vv_ds = Dataset_transformations(vv, 1, vv.shape)
#  vv_ds.normalize()
#  vv = vv_ds._items.T
#  # Append
#  _VV.append(_VV)
#  print ('VV done')

_GHT = np.array(_GHT)
_GHT = _GHT.reshape(_GHT.shape[1], _GHT.shape[0], _GHT.shape[2])
print _GHT.shape

#  _UU = np.array(_UU)
#  _UU = _UU.reshape(_UU.shape[1], _UU.shape[0], _UU.shape[2])
#
#  _VV = np.array(_VV)
#  _VV = _VV.reshape(_VV.shape[1], _VV.shape[0], _VV.shape[2])

np.savez_compressed('GHT_all.npz', G5=_GHT[:, 0, :], G7=_GHT[:, 1, :],
                    G9=_GHT[:, 2, :])
#  np.save('UU_all.npy', _UU)
#  np.save('VV_all.npy', _VV)
