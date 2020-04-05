import numpy as np
from sklearn.preprocessing import StandardScaler
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import MinMaxScaler


GHT=[]
UU=[]
VV=[]
GHT_test=[]
UU_test=[]
VV_test=[]

scalers={}
minmax_GHT={}
minmax_GHT_rest={}
for i in range(3):
	    scalers[i] = StandardScaler()
	    minmax_GHT[i]=MinMaxScaler(feature_range=(0, 1))
	    minmax_GHT_rest[i]=MinMaxScaler(feature_range=(-1, 1))

onlyfiles = [f for f in listdir('../npz/') if isfile(join('../npz/', f))]
print(onlyfiles)
for i in onlyfiles:
	if str(i)!='1994.npz':
		if str(i)!='1995.npz':
			print("extracting:",i)
			i='../npz/'+i
			year=np.load(i)
			print("UU",year['UU'].shape)
			uu=year['UU']
			uu=uu[:,:,:,0:64]

			vv=year['VV']
			print("VV",year['VV'].shape)
			vv=vv[:,:,0:64,:]

			uu=uu.reshape(uu.shape[0],uu.shape[1],uu.shape[2]*uu.shape[3])
			vv=vv.reshape(vv.shape[0],vv.shape[1],vv.shape[2]*vv.shape[3])
			print("GHT",year['GHT'].shape)
			ght=year['GHT'].reshape(year['GHT'].shape[0],year['GHT'].shape[1],year['GHT'].shape[2]*year['GHT'].shape[3])

			
			

			for i in range(ght.shape[1]):
			    ght[:, i, :] = scalers[i].fit_transform(ght[:, i, :])
			    uu[:, i, :] = scalers[i].fit_transform(uu[:, i, :])
			    vv[:, i, :] = scalers[i].fit_transform(vv[:, i, :])  

			for i in ght:
				GHT.append(i)

			for i in uu:
				UU.append(i)

			for i in vv:
				VV.append(i)
		else:
			print("extracting_test:",i)
			i='../npz/'+i
			year=np.load(i)
			print("UU",year['UU'].shape)
			print("VV",year['VV'].shape)
			print("GHT",year['GHT'].shape)

			uu=year['UU']
			uu=uu[:,:,:,0:64]

			vv=year['VV']
			vv=vv[:,:,0:64,:]

			uu=uu.reshape(uu.shape[0],uu.shape[1],uu.shape[2]*uu.shape[3])
			vv=vv.reshape(vv.shape[0],vv.shape[1],vv.shape[2]*vv.shape[3])
			ght=year['GHT'].reshape(year['GHT'].shape[0],year['GHT'].shape[1],year['GHT'].shape[2]*year['GHT'].shape[3])

			
			

			for i in range(ght.shape[1]):
			    ght[:, i, :] = scalers[i].fit_transform(ght[:, i, :])
			    uu[:, i, :] = scalers[i].fit_transform(uu[:, i, :])
			    vv[:, i, :] = scalers[i].fit_transform(vv[:, i, :])  

			for i in ght:
				GHT_test.append(i)

			for i in uu:
				UU_test.append(i)

			for i in vv:
				VV_test.append(i)
	else:
			print("extracting_test:",i)
			i='../npz/'+i
			year=np.load(i)

			uu=year['UU']
			uu=uu[:,:,:,0:64]

			vv=year['VV']
			vv=vv[:,:,0:64,:]

			uu=uu.reshape(uu.shape[0],uu.shape[1],uu.shape[2]*uu.shape[3])
			vv=vv.reshape(vv.shape[0],vv.shape[1],vv.shape[2]*vv.shape[3])
			ght=year['GHT'].reshape(year['GHT'].shape[0],year['GHT'].shape[1],year['GHT'].shape[2]*year['GHT'].shape[3])

			
			

			for i in range(ght.shape[1]):
			    ght[:, i, :] = scalers[i].fit_transform(ght[:, i, :])
			    uu[:, i, :] = scalers[i].fit_transform(uu[:, i, :])
			    vv[:, i, :] = scalers[i].fit_transform(vv[:, i, :])  

			for i in ght:
				GHT_test.append(i)

			for i in uu:
				UU_test.append(i)

			for i in vv:
				VV_test.append(i)


GHT=np.array(GHT)
VV=np.array(VV)
UU=np.array(UU)

# np.save('GHT_all.npy',GHT)
# np.save('VV_all.npy',VV)
# np.save('UU_all.npy',UU) 


GHT_test=np.array(GHT_test)
VV_test=np.array(VV_test)
UU_test=np.array(UU_test)
# np.save('GHT_test.npy',GHT_test)
# np.save('VV_test.npy',VV_test)
# np.save('UU_test.npy',UU_test) 


print("ght_test:",GHT_test.shape,GHT_test.min(),GHT_test.max(),GHT_test.mean())
print("VV_test:",VV_test.shape,VV_test.min(),VV_test.max(),VV_test.mean())
print("UU_test:",UU_test.shape,UU_test.min(),UU_test.max(),UU_test.mean())


print("ght:",GHT.shape,GHT.min(),GHT.max(),GHT.mean())
print("VV:",VV.shape,VV.min(),VV.max(),VV.mean())
print("UU:",UU.shape,UU.min(),UU.max(),UU.mean())
