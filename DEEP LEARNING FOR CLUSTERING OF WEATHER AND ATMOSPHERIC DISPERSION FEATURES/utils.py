import numpy as np
import getopt
import sys
import math
import pydot
from numpy import array
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
import logging
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils import plot_model
import matplotlib
from noise import *
from transforms3d.axangles import axangle2mat
from math import sqrt
import os
import collections 
import math
import os
import netCDF4 as netcdf4
from vis import plot_history,print_figs,plot_cluster_accuracy
from numpy import random

def load_dataset(dataset,epochs,start):
	##dataset= all pressures [:,3,:]
	##epochs
	##to select reshape for the model
	#s=0 super_stacked all pressures
	#s=3 convlstm
	#s=4 no noise 1 pressure
	#s=1 noisy 1 pressure
	##returns train-> epoch size array with the dataset



	print("loading dataset...")
	new=list()
	X = dataset
	train=list()
	train1=list()
	train2=list()
	if start==3:
		print('all Pressures...',X.shape)
		print("Presure 500:"," Mean",X[:,0,:].mean(),X[:,0,:].shape)
		print("Presure 700:"," Mean",X[:,1,:].mean(),X[:,1,:].shape)
		print("Presure 900:"," Mean",X[:,2,:].mean(),X[:,2,:].shape)
		# X=X.reshape((X.shape[0],X.shape[1]*X.shape[2]))
		# for i in range(epochs):
		# 	train.append(X)
		return X[:,1,:]

		# return train


	
	if start==0:
		## all pressures
		# print("Presure 500:"," Mean",X[:,0,:].mean(),X[:,0,:].shape)
		# print("Presure 700:"," Mean",X[:,1,:].mean(),X[:,1,:].shape)
		# print("Presure 900:"," Mean",X[:,2,:].mean(),X[:,2,:].shape)
		return X[:,0,:],X[:,1,:],X[:,2,:]

	if start==4: # no noisy simple 1 pressure
		for i in range(epochs):
			train.append(X)
		return X,train

	paths='npyz_scale/'
	print("standarized dataset")
	if os.path.isfile(paths+'random_interpolation.npy'):
		X_random_interpolated=np.load(paths+"random_interpolation.npy")
		print("X_random_interpolated",X_random_interpolated.mean(),X_random_interpolated.shape)
	else:
		X_random_interpolated=random_interpolation(X,20)
		print("X_random_interpolated",X_random_interpolated.mean(),X_random_interpolated.shape)


	if os.path.isfile(paths+'interpolation.npy'):
		X_interpolated=np.load(paths+"interpolation.npy")
		print("X_interpolated",X_interpolated.mean(),X_interpolated.shape)
	else:
		X_interpolated=interpolation(X)
		print("X_interpolated",X_interpolated.mean(),X_interpolated.shape)


	if os.path.isfile(paths+'zero_interpolation.npy'):
		X_zero=np.load(paths+"zero_interpolation.npy")
		print("X_zero",X_zero.mean(),X_zero.shape)

	else:
		X_zero=zero_interpolation(X,0.5)
		print("zero_interpolation:",X_zero.mean(),X_zero.shape)

	if os.path.isfile(paths+'X_noise.npy'):
		X_noise=np.load(paths+"X_noise.npy")
		print("gauss_noise:",X_noise.mean(),X_noise.shape)
	else:
		X_noise=gauss_noise(X)
		print("gauss_noise:",X_noise.mean(),X_noise.shape)


	X=[X,X_interpolated,X_noise,X_random_interpolated]

	np_train=list()
	train=list()
	if os.path.isfile('np_train.npy'): 
		times=np.load("np_train.npy")
		if epochs>times.shape[0]:
			b=math.ceil(epochs/times.shape[0])
			for _ in range(b):
				for j in times:
					train.append(X[j])
		elif epochs<times.shape[0]:
			times=list(times)
			for i in range(epochs):
				train.append(X[times[i]])
		else:
			for t in times:
				train.append(X[t])
		print("train in:", len(train[:epochs]))
		return X,train[:epochs]


	while len(train) <= epochs:
		print("put original")
		train.append(X[0])
		np_train.append(0)
		rand=random.choice([1,5,4,2])
		if len(train) + rand >= epochs:
			rand= epochs - len(train)+1
		x=np.random.choice(len(X))
		print("put--->",x, "times ", rand)
		for i in range(rand):
			train.append(X[x])
			np_train.append(x)
		print(len(train))

	np.save("np_train.npy",np.array(np_train))

	return X,train



def Reshape_LSTM(data):
	j=0
	#data = data[:, 1]
	#print("shape:",data.shape)
	until = int(len(data)/4)
	#print("until:----->",until)
	#4000 for X
	#for i in range(1,until,1):
	#	if len(data)%i==0:
	#		j=i
	#data = data[:, 1]
	#print("ok",j)
	n=len(data)
	#print(j)
	##2,5 days need to divide with 2009 , need 6 days divide with 574
	samples = list()
	length =int( len(data) / 574 )
	print(length)
	# step over the 5,000 in jumps of 200
	for i in range(0,n,length):
                # grab from i to i + 200
		sample = data[i:i+length]
		samples.append(sample)
	print(len(samples))
	data = array(samples)
	print(data.shape)
	return data


def overlap(data,win,t=8,silence="true"):
	#win shows the overlap px t=win no overlap , t=8 win=4 ana 4 overlap
	##simply sliding window 
	##pare tis X -> X+t, X+t -> X+(t+2) => sliding windows without overlap use win , with overlap 28/2 

	X, y = list(), list()
	in_start = 0
	n_input=t #2 days 
	n_out=t # 2 days

	fin=0
	fin1=0
	for _ in range(len(data)):
		in_end = in_start + n_input
		out_end = in_end + n_out
		if out_end <=len(data): ## could +1 or out_end <= len(data)
			# print("[",in_start,in_end,"]","[",in_end,out_end,"]")

			X.append(data[in_start:in_end, :])
			y.append(data[in_end:out_end, :]) # all features


			fin=in_end ## for the last
			fin1=out_end ##for the last

		in_start+=win ## overlap windows
	if win == 8 or win==4:
		# print(fin,fin1)
		X.append(data[fin:fin1, :])
	if silence=="true":
		print("Window shape --->",np.array(X).shape,np.array(y).shape)

	return np.array(X),np.array(y),0


def overlap_simple(data,win,t=8,silence="true"):
	#win shows the overlap px t=win no overlap , t=8 win=4 ana 4 overlap
	##simply sliding window 
	##pare tis X -> X+t, X+t -> X+(t+2) => sliding windows without overlap use win , with overlap 28/2 
	#data = X_train.reshape((X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
	rest=data
	test=list()
	if data.shape[0] % t != 0:
		

		data=data[:t*int((data.shape[0]/t))]
		if silence=="true":
			print("Window for every ",int(t/4), "day/s",data.shape)
		test=np.array(rest[data.shape[0]:rest.shape[0]])
		#print("test...",test.shape)
		test=test.reshape(1,test.shape[0],test.shape[1])
		#print("interpolate test..",test.shape)
		test=randomize_augmentation(test,target=win)
		#print("run perm.. test")
		# for i in range(47):
		# 	test.append(np.random.permutation(test))
		#print("rest for test...",np.array(test).shape)

	X, y = list(), list()
	in_start = 0
	n_input=t #2 days
	n_out=t
	sys.path.insert(1,'../wrfhy/wrfvol')
	from netcdf_subset import netCDF_subset as ns
	os.chdir("../wrfhy/wrfvol")
	print(os.getcwd())
	data1 = ns('40years.nc', [500,700,900], ['GHT'])
	data1._time_name = 'Times'

	for _ in range(len(data)):
		in_end = in_start + n_input
		if in_end <len(data): ## could +1 or out_end <= len(data)
			#print("[",in_start,in_end,"]","[",in_end,out_end,"]")

			X.append(data[in_start:in_end, :])
			

			name = 'exp_cluster_'+str(_)+'.nc'
			data1.exact_copy_file(name, [in_start,in_end])
			info=netcdf4.Dataset(name)
			for i in info['Times'][:]:
				print(i.tostring())
				if  "06:00:00" in i.tostring():
					print("[",in_start,in_end,"]","[",in_end,out_end,"]")
					sys.exit(2)

			print("-----------------------------------------")

			#y.append(data[in_end:out_end, 0]) ## predict next point
		in_start+=win
	if silence=="true":
		print("Window shape --->",np.array(X).shape,np.array(y).shape)

	return np.array(X),np.array(y),np.array(test)


def window_reverse_y(X_train):
	#weak1-> reverse(week1) , no overlap
	#data = X_train.reshape((X_train.shape[0]*X_train.shape[1], X_train.shape[2]))
	in_start = 0 
	win=8
	X, y = list(), list()
	for i in range(len(X_train)):
		out=in_start + win 
		if in_start < len(X_train): 
			print("[",in_start,out,"]")
			X.append(X_train[in_start:out,:])
			y.append(np.flip(X_train[in_start:out,:])) 
		in_start=out 
	return np.array(X),np.array(y)

def randomize_augmentation(X,target=48):
	import random
	X_rand=list()
	for i in X:
		d=list(i)
		while len(d)< target:
			w=random.randint(0,len(d)-1)
			d=d[:w]+[d[w]]+d[w:]
		X_rand.append(d)
	print("Augmented Window shape --->",np.array(X_rand).shape)
	return np.array(X_rand)


def load_lstm_trained_model(weights_path,n_in,d,activation,units):
	model = LSTM_ae(n_in,d,activation,units)
	model.load_weights(weights_path)
	return model


def forecast(model, history, n_input):
	# flatten data
	data = array(history)
	data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
	print(input_x.shape)
	# retrieve last observations for input data
	input_x = data[-n_input:, :]
	# reshape into [1, n_input, n]
	input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))
	# forecast the next week
	yhat = model.predict(input_x)
	# we only want the vector forecast
	yhat = yhat[0]
	return yhat



def clustering(yhat,type,n_clusters=30):
	print("clustering...")
	print("Encoded Dimensions clustering with shape ",yhat.shape," and mean ",yhat.mean())
	print("yhat---->",yhat)
	
	#yhat = yhat.reshape((yhat.shape[0]*yhat.shape[1], yhat.shape[2]))
	#print("Reshape before clustering..",yhat.shape)
	scores_calinski=[]
	scores_silhouette=[]
	#k-means
	for i in range(2,n_clusters,1):
	        kmeans = KMeans(n_clusters=i,n_jobs=-1,n_init=20)
	        pred_kmeans = kmeans.fit_predict(yhat)
	        if len(set(kmeans.labels_))==1:
	        	print("cant cluster... all cluster same")
	        	return

	        print(kmeans.labels_)
	        cnt = collections.Counter()
	        for l in kmeans.labels_:
	        	cnt[l]+=1
	        print("Count Labels:",cnt)

	        b=metrics.calinski_harabaz_score(yhat,pred_kmeans)
	        scores_calinski.append(b)
	        print('calinski k-means cluster: ',i,' ',b)

	        a=metrics.silhouette_score(yhat, pred_kmeans, metric='euclidean')
	        scores_silhouette.append(a)
	        ## cluster measures##
	        print('silhouete k-means cluster: ',i,' ',a)
	        #print(type,'silhouete k_means cluster: ',i,' ',metrics.silhouette_score(yhat, pred_kmeans, metric='euclidean'))
	plot_cluster_accuracy(scores_silhouette,type,"silhouete_kmeans")
	plot_cluster_accuracy(scores_calinski,type,"calinski_kmeans")


def pad_seq(X_train,X_test):
	npad = ((1, 0), (0, 0), (0, 0))
	X_train= np.pad(X_train, pad_width=npad, mode='constant', constant_values=0)
	X_test= np.pad(X_test, pad_width=npad, mode='constant', constant_values=0)
	return X_train,X_test



def generator(features, batch_size,d):
	print(d)
	'''
	features: (X,64,64,3)
	batch_size = size of batch examples 
	d = channels 
	'''
	# Create empty arrays to contain batch of features and labels#
	batch_features = np.zeros((batch_size,8, 64, 64, d))

	while True:
	   for i in range(batch_size):
	     # choose random index in features
	     index= random.choice( (len(features)) ,1 )
	     batch_features[i] = features[index]
	   yield batch_features,batch_features


