import numpy as np
import tensorflow
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Input, Add,Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape,Conv2DTranspose,RepeatVector,LSTM,TimeDistributed, Conv1D, GlobalAveragePooling1D , MaxPooling1D , Dropout, UpSampling1D, ConvLSTM2D, Flatten, Dense, Dropout, Lambda, Concatenate, Average, Bidirectional
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.callbacks import CSVLogger, EarlyStopping, ModelCheckpoint, TerminateOnNaN, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.python.keras.initializers import VarianceScaling,glorot_uniform, glorot_normal
from sklearn.cluster import KMeans
import getopt
import sys
import math
import pydot
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras.optimizers import Adam,RMSprop,Adagrad,SGD
from tensorflow.python.keras.utils import plot_model
from numpy import array
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from matplotlib import pyplot as plt
from tensorflow.python.keras.utils import plot_model
import matplotlib
from scipy.interpolate import CubicSpline
from noise import *
from transforms3d.axangles import axangle2mat
from math import sqrt
import os
import collections 
from utils import load_dataset,overlap,window_reverse_y,clustering, generator
from vis import plot_history,print_figs,plot_cluster_accuracy
from tensorflow.python.keras import losses
from vis import plot_tsne
from sklearn.manifold import TSNE
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from tsne import closest
import math
#parameters
epochs=50
#sgd_lr=0.0001 -> works , 0.0005
sgd_lr=0.001 
l_activation='relu'
nl_activation= 'linear'
Adam=Adam(lr=sgd_lr)
RMSprop=RMSprop(lr= sgd_lr)
Sgd=SGD(lr=sgd_lr,momentum=0.99, nesterov=True) ## worked with 0.9ls , momentum=0.9 , nesterov=True
optimizer=Sgd
loss='mean_squared_error'
batch_size=256
units=32
clusters=15
optimizerss=[Sgd]
##simple ae

def rmse(y_actual, y_predicted):
	return K.sqrt(K.mean(K.square(y_actual - y_predicted))) 


def train_ae(autoencoder,optimizer,loss,X,batch_size,epochs,sgd_lr,reverse_reconstruction,flag=0,noise="false",reset="false"):
	#train ae
  type="lstm_plain_"+str(units)+str(optimizer)
  print(type," train ae for:",epochs,"optimizer: ",optimizer,"loss: ", loss,'batch_size: ',batch_size)
  autoencoder.compile(optimizer=optimizer, loss=loss)
    #plot_model(autoencoder, to_file=type, show_shapes=True, show_layer_names=True)

  if noise=="true":
  	type+="_noise"
  if reset=="true":
  	type+="_stateful"
  	historyy=[]
  	losss=[]
  	t1=[]
  if reverse_reconstruction=="true":
    type+="_reverse_reconstruction"
  if flag==1:
  	type+="_stacked_"


  historyy=[]
  losss=[]

  filepath='tmp/' + type + '.hdf5'
  log_dir='./logs_'+type
  tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
  #reduce_lr = ReduceLROnPlateau(factor=0.2,patience=10, min_lr=0.0001,verbose=1)
  early=EarlyStopping(monitor='loss', min_delta=0, patience=20, verbose=0, mode='auto')

  checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
  K.set_value(optimizer.lr,sgd_lr)
  print("------------>Starting LR :",K.get_value(autoencoder.optimizer.lr))


  for e,i in enumerate(X):
    print("epoch..",e,"/",epochs)
    print(i.shape)
    X_train,x_test,test=overlap(i,4,t=8)
    # if e!=0 and e==20 and e!=epochs:
    print("------------>LR BEFORE:",K.get_value(autoencoder.optimizer.lr))
    K.set_value(optimizer.lr,sgd_lr)
    print("------------>LR NOW:",K.get_value(autoencoder.optimizer.lr))

    # if e!=0 and e==40 and e!=epochs:
    # 	print("------------>LR BEFORE:",K.get_value(autoencoder.optimizer.lr))
    # 	K.set_value(optimizer.lr,  K.get_value(optimizer.lr) / 10)
    # 	print("------------>LR NOW:",K.get_value(autoencoder.optimizer.lr))
    	
   # model.lr.set_value(.02)
    if reverse_reconstruction == "true":
        ##reverse reconstruction
        x_train=np.flip(X_train) ##::-1
    else:
        x_train=X_train
    if reset!="true":
      #no reset state
    	print('Shuffle true')
    	history=autoencoder.fit(X_train, x_train,epochs=1,batch_size=batch_size,shuffle=True,callbacks=[tbCallback,checkpointer,TerminateOnNaN(),early])
    if reset=="true":
    	print('Shuffle false')
    	history=autoencoder.fit(X_train, x_train,epochs=1,batch_size=batch_size,shuffle=False,callbacks=[tbCallback,checkpointer,TerminateOnNaN(),early])

    	autoencoder.reset_states()

    losss.append(history.history['loss'][0])
  historyy.append(losss)
  

  if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
    return
  if reset=="true":
    plot_history(historyy,type)
  else:
   plot_history(historyy,type)

  X_train,x_test,test=overlap(np.array(X[0]),4,t=8) ## to allaksa
  seq_in=X_train
  print("predict in shape....",seq_in.shape)

  if reset!="true":
       yhat=autoencoder.predict(seq_in)
  else:
        yhat=autoencoder.predict(seq_in,batch_size=batch_size)
  if reverse_reconstruction == "true":
        yhat=np.flip(yhat)

  print("---->",yhat.shape,seq_in.shape)
  print_figs(seq_in,yhat,3,type)

  #clustering the encodings only
  #print(autoencoder.layers)
  model = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer("encoder").output)
  if flag ==1:
  	  model = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer("encoder").output)

  if reset=="true":
        yhat = model.predict(seq_in,batch_size=batch_size)
  else:
        yhat = model.predict(seq_in)

  print("previous Mean..",seq_in.mean())
  print("YHAT-------->",yhat)
  savp=type+"_representation.npy"
  np.save(savp,yhat)
  # clustering(yhat,type)




def model_conv_lstm(timesteps):
	visible  = Input(shape=(timesteps, 64, 64, 1))

	x = ConvLSTM2D(filters=64, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   ,return_sequences=True
	                   , activation='relu'
	                   , padding='same')(visible)

	# x = ConvLSTM2D(filters=64, kernel_size=(3, 3)
	#                    , data_format='channels_last'
	#                    ,return_sequences=True
	#                    , activation='relu'
	#                    , padding='same')(x)

	# x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
 #                   , data_format='channels_last'
 #                   ,return_sequences=True
 #                   , activation='relu'
 #                   , padding='same')(x)

	# x=TimeDistributed(MaxPooling2D((2,2)))(x)

	x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   ,return_sequences=True
	                   , activation='relu'
	                   , padding='same')(x)

	#x= BatchNormalization()(x)
	# x=Dropout(0.2)(x)


	x=TimeDistributed(MaxPooling2D((2,2)))(x)

	# x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
	#                    , data_format='channels_last'
	#                    ,return_sequences=True
	#                    , activation='relu'
	#                    , padding='same')(x)

	# x=TimeDistributed(MaxPooling2D((2,2)))(x)

	x = ConvLSTM2D(filters=16, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   ,return_sequences=True
	                   , activation='relu'
	                   , padding='same')(x)

	x=TimeDistributed(MaxPooling2D((2,2)))(x)

	x = ConvLSTM2D(filters=8, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   ,return_sequences=True
	                   , activation='relu'
	                   , padding='same')(x)



	x = ConvLSTM2D(filters=8, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   ,return_sequences=True
	                   , activation='relu'
	                   , padding='same',name="encoder")(x)


	x=TimeDistributed(MaxPooling2D((2,2)))(x)

	# encoded=TimeDistributed(Dense(32,),name='dense')(encoded)
	pre_flat_shape = encoded.shape[1:].as_list()
	x=TimeDistributed(flatten())(x)

	x=Dense(8,name='dense1',activation='relu')(x)  
	encoded=Dense(4,name='dense2',activation='linear')(x)  

	# x=TimeDistributed(flatten())(x)
	#x=TimeDistributed(MaxPooling2D((2,2),(2,2)))(x)
	#encoder=Model(visible,encoded)

	# x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
	#                    , data_format='channels_last'
	#                    , activation='relu'	                  
	#                    , padding='same',return_sequences=True)(x)

	
	# x = BatchNormalization()(x)
	#x=TimeDistributed(Dense(pixels_x*pixels_y*channels))(x)
	pre_flat_shape[0] = 8
	x = Dense(np.product(pre_flat_shape), activation='relu')(encoded)
	x = Reshape(pre_flat_shape)(x)

	x=TimeDistributed(UpSampling2D((2,2)))(x)

	x = ConvLSTM2D(filters=8, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   , activation='relu'	                  
	                   , padding='same',return_sequences=True)(x)

	x = ConvLSTM2D(filters=8, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   , activation='relu'	                  
	                   , padding='same',return_sequences=True)(x)

	# x = BatchNormalization()(x)

	x=TimeDistributed(UpSampling2D((2,2)))(x)

	x = ConvLSTM2D(filters=16, kernel_size=(3, 3)
                   , data_format='channels_last'
                   , activation='relu'	                  
                   , padding='same',return_sequences=True)(x)

	x=TimeDistributed(UpSampling2D((2,2)))(x)

	x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
	                   , data_format='channels_last'
	                   ,return_sequences=True
	                   , activation='relu'
	                   , padding='same')(x)


	# x = ConvLSTM2D(filters=8, kernel_size=(3, 3)
	#                    , data_format='channels_last'

	#                    , activation='relu'
	#                    , padding='same',return_sequences=True)(x)
	# x= BatchNormalization()(x)
	# x=TimeDistributed(UpSampling2D((2,2)))(x)
	
	# x = ConvLSTM2D(filters=64, kernel_size=(3, 3)
	#                    , data_format='channels_last'
	#                    , activation='relu'
	#                    , padding='same',return_sequences=True)(x)

	# x=TimeDistributed(UpSampling2D((2,2)))(x)


	# x = ConvLSTM2D(filters=32, kernel_size=(3, 3)
	#                    , data_format='channels_last'
	#                    ,return_sequences=True
	#                    , activation='relu'
	#                    , padding='same')(x)

	x = ConvLSTM2D(filters=64, kernel_size=(3, 3)
                   , data_format='channels_last'
                   ,return_sequences=True
                   , activation='relu'
                   , padding='same')(x)

	# x=TimeDistributed(UpSampling2D((2,2)))(x)

	# x = ConvLSTM2D(filters=64, kernel_size=(3, 3)
	#                    , data_format='channels_last'
	#                    , activation='relu'
	#                    , padding='same',return_sequences=True)(x)

                  
	decoded=TimeDistributed(Conv2D(1,kernel_size=(3,3),padding="same",activation='linear'))(x)
	model=Model(inputs=visible,outputs=decoded)

	print(model.summary())


	return model



	
def conv_lstm(X,batch_size,optimizer,sgd_lr):
	dropout=0.2
	r=0.001
	seq_in,seq_out,_=overlap(X,4,t=8)
	seq_in = seq_in.reshape((seq_in.shape[0], seq_in.shape[1], 64, 64, 1))
	# seq_out = seq_out.reshape((seq_out.shape[0], seq_out.shape[1], 64, 64, 1))

	model = model_conv_lstm(seq_in.shape[1]) 

	print(model.summary())

	# plot_model(model, show_shapes=True, to_file="conv_lstm.png")



	historyy=[]
	losss=[]
	type="conv_lstm_"+str(optimizer)
	filepath='tmp/' + type + '.hdf5'
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')

	
	#reduce_lr = ReduceLROnPlateau(factor=0.2,patience=10, min_lr=0.0001,verbose=1)

	model.compile(loss=loss,optimizer=optimizer)
	print(model.summary())

	print("------------>Starting LR :",K.get_value(model.optimizer.lr))
	K.set_value(optimizer.lr,  sgd_lr)
	print("------------>LR NOW:",K.get_value(model.optimizer.lr))


	history=model.fit(seq_in,seq_in,epochs=15,batch_size=batch_size,callbacks=[checkpointer])
	# history=model.fit_generator(generator(seq_in,batch_size,seq_in.shape[4]),steps_per_epoch=math.ceil(seq_in.shape[0] / batch_size),verbose=1,epochs=epochs,callbacks=[checkpointer])
	losss=[history.history['loss']]
	plot_history(losss,type)
	# for e,i in enumerate(X):
	# 	print("epoch..",e,"/",epochs)
	# 	print(i.shape)
	# 	X_train,x_test,_=overlap(i,4,t=8)
	# 	X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 64, 64, 3))
	# 	history=model.fit(X_train,X_train, epochs=1, batch_size=batch_size,shuffle=True)
	# 	losss.append(history.history['loss'][0])
	# historyy.append(losss)

	# plot_history(historyy,"conv_lstm")
	#print_figs(seq_in,yhat,3,"conv_lstm")
	
	# encoder = Model(inputs=model.input, outputs=model.get_layer('encoder').output)
	# yhat=encoder.predict(seq_in)
	# print("---!!!!!->",yhat.shape)
	# yhat = yhat.reshape((yhat.shape[0]*seq_in.shape[1], 8*8*2))
	# savp=type+"_representation.npy"
	# np.save(savp,yhat)
	# clustering(yhat,"conv_lstm")






def stacked_lstm_ae(timesteps,n_features,activation,units,optimizer,dropout):
  # print("shape:---> ",timesteps,n_features)
  #dropout=dropout
  lstm_autoencoder = Sequential()
  # r=reg #works with 0.01
  # if units == 256:
  # 	r=0.1
  # # Encoderaaaaa
  # un=units*2
  r=0.01
  # dropout=0.7
  lstm_autoencoder.add(LSTM(2048, activation='relu',activity_regularizer=regularizers.l1(r),input_shape=(timesteps, n_features), return_sequences=True))
  lstm_autoencoder.add(LSTM(256, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(128, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(64, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(32, activation='linear',activity_regularizer=regularizers.l1(r),name="encoder"))
  lstm_autoencoder.add(RepeatVector(timesteps))
  # Decoder ,activity_regularizer=regularizers.l1(r),activity_regularizer=regularizers.l1(r),activity_regularizer=regularizers.l2(r)
  # lstm_autoencoder.add(LSTM(32, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(64, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(128, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(256, activation='relu',activity_regularizer=regularizers.l1(r),return_sequences=True))
  lstm_autoencoder.add(LSTM(2048, activation='linear',activity_regularizer=regularizers.l1(r),return_sequences=True))
  # lstm_autoencoder.add(LSTM(32, activation='relu',input_shape=(timesteps, n_features), return_sequences=True))

  lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

  lstm_autoencoder.summary()

  return lstm_autoencoder              




	    

def LSTM_ae(n_in,d,activation,units,optimizer,stateful="false",batch_size=9):
		##lstm_plain
		dropout= 0.2
		#un=int(units/2)
		# model = Sequential()
		#,dropout=dropout,activity_regularizer=regularizers.l2(0.001),
		if stateful=="true":
			visible = Input(batch_shape=(batch_size,n_in,d))
			encoder = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.01),stateful=True)(visible)
			# define reconstruct decoder
			decoder1 = RepeatVector(n_in)(encoder)
			decoder1 = LSTM(units, activation=activation, dropout=dropout,activity_regularizer=regularizers.l2(0.01),stateful=True,return_sequences=True)(decoder1)
			decoder1 = TimeDistributed(Dense(d))(decoder1)
			type="lstm_plain_stateful"
		else:
			visible = visible = Input(shape=(n_in,d))
			encoder = LSTM(units, activation=activation,dropout=dropout,name="encoder")(visible)
			decoder1 = RepeatVector(n_in)(encoder)
			decoder1 = LSTM(units, activation='linear',dropout=dropout,return_sequences=True)(decoder1)
			decoder1 = TimeDistributed(Dense(d))(decoder1)
			type="lstm_plain"
		model = Model(inputs=visible, outputs=decoder1)
		print(model.summary())

		type="lstm_ae"+str(units)+str(optimizer)+".png"
		plot_model(model, show_shapes=True, to_file=type)
		# hidden_states = K.variable(value=np.random.normal(size=(n_in,d)))
		# cell_states = K.variable(value=np.random.normal(size=(n_in,d)))

		# model.layers[1].states[0] = hidden_states
		# model.layers[1].states[1] = cell_states

		#n_out = n_in -1
        

		return model

    
    #return vae, encoder, generator

def lstm_ght_all(timesteps,n_features,lat):


  Xinput = Input(shape=(timesteps, n_features))
  Yinput = Input(shape=(timesteps, n_features))
  Zinput = Input(shape=(timesteps, n_features))

  Xencoded = LSTM(64, activation='relu',return_sequences=True)(Xinput)
  Xencoded=LSTM(32, activation='linear')(Xencoded)


  Yencoded = LSTM(64, activation='relu',return_sequences=True)(Yinput)
  Yencoded=LSTM(32, activation='linear')(Yencoded)

  Zencoded = LSTM(64, activation='relu',return_sequences=True)(Zinput)
  Zencoded=LSTM(32, activation='linear')(Zencoded)

  if lat == 'avg':
    shared_input = Average(name='avg')([Xencoded, Yencoded,Zencoded])
  else:
    shared_input = Concatenate(name='concat')([Xencoded, Yencoded,Zencoded])

  shared_output = Dense(32, activation='linear',name='last')(shared_input)
  shared_output =RepeatVector(timesteps)(shared_output)

  Xdecoded=LSTM(32, activation='relu',return_sequences=True)(shared_output)
  Xdecoded = LSTM(64, activation='linear',return_sequences=True)(Xdecoded)
  Xdecoded=TimeDistributed(Dense(n_features))(Xdecoded)

  Ydecoded=LSTM(32, activation='relu',return_sequences=True)(shared_output)
  Ydecoded = LSTM(64, activation='linear',return_sequences=True)(Ydecoded)
  Ydecoded=TimeDistributed(Dense(n_features))(Ydecoded)

  Zdecoded=LSTM(32, activation='relu',return_sequences=True)(shared_output)
  Zdecoded = LSTM(64, activation='linear',return_sequences=True)(Zdecoded)
  Zdecoded=TimeDistributed(Dense(n_features))(Zdecoded)
    
                 

  model = Model([Xinput, Yinput,Zinput], [Xdecoded, Ydecoded,Zdecoded])

  return model


    


def super_stacked(timesteps,n_features,optimizer,X_input,Y_input,Z_input,epochs,batch_size,X,X1,X2,sgd_lr,lat):
	
  print(timesteps,n_features)

  Xinput = Input(shape=(timesteps, n_features))
  Yinput = Input(shape=(timesteps, n_features))
  Zinput = Input(shape=(timesteps, n_features))

  Xencoded = LSTM(64, activation='relu',return_sequences=True)(Xinput)
  Xencoded=LSTM(32, activation='linear')(Xencoded)


  Yencoded = LSTM(64, activation='relu',return_sequences=True)(Yinput)
  Yencoded=LSTM(32, activation='linear')(Yencoded)

  Zencoded = LSTM(64, activation='relu',return_sequences=True)(Zinput)
  Zencoded=LSTM(32, activation='linear')(Zencoded)

  if lat == 'avg':
    shared_input = Average(name='avg')([Xencoded, Yencoded,Zencoded])
  else:
    shared_input = Concatenate(name='concat')([Xencoded, Yencoded,Zencoded])

  shared_output = Dense(32, activation='linear',name='last')(shared_input)
  shared_output =RepeatVector(timesteps)(shared_output)

  Xdecoded=LSTM(32, activation='relu',return_sequences=True)(shared_output)
  Xdecoded = LSTM(64, activation='linear',return_sequences=True)(Xdecoded)
  Xdecoded=TimeDistributed(Dense(n_features))(Xdecoded)

  Ydecoded=LSTM(32, activation='relu',return_sequences=True)(shared_output)
  Ydecoded = LSTM(64, activation='linear',return_sequences=True)(Ydecoded)
  Ydecoded=TimeDistributed(Dense(n_features))(Ydecoded)

  Zdecoded=LSTM(32, activation='relu',return_sequences=True)(shared_output)
  Zdecoded = LSTM(64, activation='linear',return_sequences=True)(Zdecoded)
  Zdecoded=TimeDistributed(Dense(n_features))(Zdecoded)
  	
  if lat=='avg':
    type="super_stacked_avg"+str(32)+str(optimizer)
  else:
    type="super_stacked_concat"+str(32)+str(optimizer)
  model = Model([Xinput, Yinput,Zinput], [Xdecoded, Ydecoded,Zdecoded])
  print(model.summary())
  
  # plot_model(model, show_shapes=True, to_file=typ+'.png')
  model.compile(optimizer=optimizer, loss=['mse', 'mse','mse'])
  filepath='tmp/' + type + '.hdf5'
  checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
  print("------------>Starting LR :",K.get_value(model.optimizer.lr))
  K.set_value(optimizer.lr,  sgd_lr)
  print("------------>LR NOW:",K.get_value(model.optimizer.lr))
  history=model.fit([X_input,Y_input,Z_input],[X_input,Y_input,Z_input],epochs=epochs,batch_size=batch_size,callbacks=[checkpointer])
  losss=[history.history['loss']]
  #print(losss)
 

  plot_history(losss,type)

  # model = Model(inputs=model.inputs, outputs=model.get_layer("last").output)
  

  # ##overlap 1 day 
  # X_train1,train_y,_=overlap(X,4,t=8) # 700 GHT 
  # X_train2,train_y2,_=overlap(X1,4,t=8) # 500 GHT
  # X_train3,train_y3,_=overlap(X2,4,t=8) # 900 GHT

  # yhat = model.predict([X_train1,X_train2,X_train3])


  # # print("previous Mean..",X_train1.mean(),X_train2.mean(),X_train3.mean())
  # # print("YHAT-------->",yhat)
  # # savp=type+"_representation.npy"
  # # np.save(savp,yhat)
  # # clustering(yhat,type)

def model_cnnlstm_transpose():

	'''
	Try to add LSTM in the Decoder
	'''
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='linear', padding='same'))(visible) ##added instead of relu
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

	x = Dense(16,activation='relu')(x)##added
	
	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = LSTM(512, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	x = LSTM(256, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	pre_flat_shape[0] = 8
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)

	x = Dense(16,activation='relu')(x)##added
	x = TimeDistributed(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2DTranspose(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)

	x = TimeDistributed(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2DTranspose(1, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)


	return model



def model_cnndeclstm():

	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='linear', padding='same'))(visible) ##added instead of relu
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
		
	x = Dense(16,activation='relu')(x)##added
	x = Dense(8,activation='relu')(x)##added
	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = LSTM(512, activation='relu',return_sequences=True)(x)
	x = LSTM(256, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	x = LSTM(128, name='bottleneck', activation='linear')(x)
	# x= TimeDistributed(8)
	# x = LSTM(256, activation='relu',return_sequences=True)(x) 
	# x = LSTM(512, activation='relu',return_sequences=True)(x) 
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)



	model=Model(inputs=visible,outputs=decoded)


	return model



def model_cnnlstm2(encoding=32):

	'''
	Try to add LSTM in the Decoder
	'''
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='linear', padding='same'))(visible) ##added instead of relu
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	# x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	

	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = Dense(512,activation='relu')(x)##added
	x = LSTM(128, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	x = LSTM(encoding, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	pre_flat_shape[0] = 8

	x = Dense(512,activation='relu')(x)##added
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)

	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	# x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)



	return model


def model_cnn_lstm_concat2():
	
	# CNN Encoder
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(64, (3, 3), activation='linear', padding='same'))(visible) ##added instead of relu
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	pre_flat_shape = x.shape[1:].as_list()

	x = Flatten()(x)
	x = Dense(64,activation='relu')(x)



	# LSTM Encoder

	Xinput = Input(shape=(8, 4096))

	Xencoded = LSTM(256, activation='linear',return_sequences=True)(Xinput)
	Xencoded=LSTM(256, activation='relu',return_sequences=True)(Xencoded)
	# Xencoded = TimeDistributed( Flatten() )(x)
	Xencoded =TimeDistributed( Dense(256, activation='relu') )(Xencoded)
	Xencoded = Dense(np.product(pre_flat_shape), activation='relu')(Xencoded)
	Xencoded = Reshape(pre_flat_shape)(Xencoded)

	Xencoded = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(Xencoded)
	Xencoded = Flatten()(Xencoded)
	Xencoded = Dense(64,activation='relu')(Xencoded)
	print('--------',Xencoded.shape)



	concat = Concatenate(axis=1)
	Xencoded = concat([x,Xencoded])
	x = Dense(128,activation='linear',name='bottleneck')(Xencoded)
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)

	# Decoder
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)


	model=Model(inputs=[visible,Xinput],outputs=decoded)

	return model

def model_cnn_lstm_concat():
	
	# CNN Encoder
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(visible) ##added instead of relu
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	# pre_flat_shape = x.shape[1:].as_list()

	x = Flatten()(x)
	x = Dense(128,activation='relu')(x)


	# LSTM Encoder

	Xinput = Input(shape=(8, 4096))

	Xencoded = LSTM(128, activation='relu',return_sequences=True)(Xinput)
	# Xencoded=LSTM(512, activation='relu',return_sequences=True)(Xencoded)
	Xencoded=LSTM(128, activation='relu')(Xencoded)
	concat = Concatenate(axis=1)
	concat = concat([x,Xencoded])
	# print(concat.shape)
	x = Dense(32,activation='linear',name='bottleneck')(concat)

	x = Dense(16*16*1,activation='linear')(concat)
	# x=Dropout(0.3)(x)
	x = Dense(np.product([8,16,16,1]), activation='relu')(x)

	x = Reshape((8,16,16,1))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)


	# Decoder
	x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)


	model=Model(inputs=[visible,Xinput],outputs=decoded)

	return model

def model_cnnlstm4(encoding=32):

	'''
	Try to add LSTM in the Decoder
	'''
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='linear', padding='same'))(visible) ##added instead of relu
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	# x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	

	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = Dense(1024,activation='relu')(x)##added
	x = LSTM(512, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	x = LSTM(encoding, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	pre_flat_shape[0] = 8

	x = Dense(1024,activation='relu')(x)##added
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)

	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	# x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)


	return model



def mode_cnnlstm3(encoding=32):

	'''
	Try to add LSTM in the Decoder
	'''
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu',strides=(1,1), padding='same'))(visible)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu',strides=(1,1), padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3),strides=(2,2), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), strides=(2,2),activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3),strides=(2,2), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3),strides=(2,2), activation='relu', padding='same'))(x)

	# x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = LSTM(256, name='Bottleneck' ,activation='linear')(x)

	### x = LSTM(16*16*1, activation='linear',return_sequences=False)(x)

	# Bottleneck here!
	# x = LSTM(encoding, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	
	# x = Dense(16*16*1, name='Bottleneck',activation='linear')(x)
	# x=Dropout(0.2)(x)
	x = Dense(np.product([8,1,1,64]), activation='relu')(x)
	

	x = Reshape((8,8,8,1))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3),activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)

	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)

	return model


def model_cnnlstm(encoding=32):

	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), strides=(1,1),activation='relu', padding='same'))(visible)
	# x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3),strides=(1,1),activation='relu', padding='same'))(x)

	x = TimeDistributed(MaxPooling2D((2, 2),padding='same'))(x)

	# x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3),strides=(1,1),activation='relu', padding='same'))(x)

	x = TimeDistributed(MaxPooling2D((2, 2),padding='same'))(x)

	x = TimeDistributed(Conv2D(16, (3, 3),strides=(1,1),activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), strides=(1,1),activation='relu', padding='same'))(x)

	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	# x = Reshape(pre_flat_shape)(x)
	# x = Dense(512,activation='relu')
	# x = Dense(128,activation='relu')

	x = LSTM(512, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	# x = LSTM(64, activation='relu')(x)
	x = LSTM(encoding, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	pre_flat_shape[0] = 8
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)
	
	x = TimeDistributed(Conv2D(16, (3, 3), strides=(1,1),activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3) ,strides=(1,1),activation='relu', padding='same'))(x)

	x = TimeDistributed(UpSampling2D((2, 2)))(x)

	# x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3),strides=(1,1), activation='relu', padding='same'))(x)

	x = TimeDistributed(UpSampling2D((2, 2)))(x)

	x = TimeDistributed(Conv2D(32, (3, 3),strides=(1,1),activation='relu', padding='same'))(x)
	# x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3),strides=(1,1),activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3),activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)


	return model

def cnnlstm(X,batch_size,optimizer,sgd_lr,model,encoding_dim=32,lstm='no'):

	seq_in,_,_=overlap(X,4,t=8)
	if lstm=='yes':
		d,_,_=overlap(X,4,t=8)

	seq_in = seq_in.reshape((seq_in.shape[0], seq_in.shape[1], 64, 64, 3))

	dataset=[]
	for i in range(epochs+1):
		dataset.append(X)

	# model =model_cnnlstm()


	historyy=[]
	losss=[]
	type="cnn_lstm_"+str(optimizer)
	filepath='tmp/' + type + '.hdf5'
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')

	#reduce_lr = ReduceLROnPlateau(factor=0.2,patience=10, min_lr=0.0001,verbose=1)

	model.compile(loss=loss,optimizer=optimizer)
	
	K.set_value(optimizer.lr,sgd_lr)
	print("------------>Starting LR :",K.get_value(model.optimizer.lr))

	print(model.summary())
	if lstm=='yes':
		history=model.fit([seq_in,d],[seq_in],epochs=800,batch_size=batch_size,callbacks=[checkpointer])

	else:
		# for e,i in enumerate(dataset):
		# 	seq_in,_,_=overlap(i,4,t=8)
		# 	seq_in = seq_in.reshape((seq_in.shape[0], seq_in.shape[1], 64, 64, 1))


		# 	if e % 50==0 and e>0: 
		# 		print("------------>was LR :",K.get_value(model.optimizer.lr))
		# 		K.set_value(optimizer.lr,  sgd_lr/10)
		# 		print("------------>LR NOW:",K.get_value(model.optimizer.lr))
		history=model.fit(seq_in,seq_in,epochs=epochs,batch_size=batch_size,callbacks=[checkpointer])
	# history=model.fit_generator(generator(seq_in,batch_size,seq_in.shape[4]),steps_per_epoch=math.ceil(seq_in.shape[0] / batch_size),verbose=1,epochs=epochs,callbacks=[checkpointer])
	losss=[history.history['loss']]
	plot_history(losss,type)


def compose_model():
	## each conv for a 1 6hour slot.

	pass

def model_cnnlstm1():

	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(visible)
	# x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)

	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

	# x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)

	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)

	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)

	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = LSTM(512, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	x = LSTM(64, activation='relu')(x)
	x = Dense(32, name='bottleneck', activation='linear')(x)

	# Start scaling back up
	# No frame stack for output
	pre_flat_shape[0] = 8
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)
	
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)

	x = TimeDistributed(UpSampling2D((2, 2)))(x)

	# x = TimeDistributed(Conv2D(8, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)

	x = TimeDistributed(UpSampling2D((2, 2)))(x)

	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	# x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)
	# x =TimeDistributed(Flatten())(x)

	# x = TimeDistributed(UpSampling2D((2,2)))(x)
	# x= TimeDistributed( Conv2D(3, (2,2), activation='relu', padding='same') )(x)
	# x= TimeDistributed( Conv2D(3, (2,2), activation='relu', padding='same') )(x)
	# x =TimeDistributed(UpSampling2D((2, 2))) (x)
	# x= TimeDistributed( Conv2D(3, (2,2), activation='relu', padding='same') )(x)
	# x = TimeDistributed( Conv2D(3, (2,2), activation='relu', padding='same') )(x)
	# x =TimeDistributed(UpSampling2D((2, 2))) (x)
	# x= TimeDistributed( Conv2D(3, (2,2), activation='relu', padding='same') )(x)
	# x= TimeDistributed( Conv2D(3, (2,2), activation='relu', padding='same') )(x)

	# cnn.add(LSTM(32, activation='relu',return_sequences=True))
	# cnn.add(LSTM(64, activation='relu',return_sequences=True))
	# cnn.add(Flatten())
	# cnn.add(MaxPooling2D(pool_size=(2, 2)))
	# cnn.add(Conv2D(1, (2,2), activation='relu', padding='same'))

	# model=Model(inputs=visible,outputs=x)

	# print(model.summary())
	return model



def cnnlstm1(X,batch_size,optimizer,sgd_lr,encoding_dim=32):

	# seq_in,seq_out,_=overlap(X,4,t=8)
	# seq_in = seq_in.reshape((seq_in.shape[0], seq_in.shape[1], 64, 64, 1))

	# XX = []
	# for i in range(epochs):
	# 		XX.append(X)

	# print(len(XX))

	model=model_cnnlstm1()



	historyy=[]
	losss=[]
	type="cnn1_lstm_"+str(optimizer)
	filepath='tmp/' + type + '.hdf5'
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')


	model.compile(loss=loss,optimizer=optimizer)
	print("------------> LR :",K.get_value(model.optimizer.lr))
	K.set_value(model.optimizer.lr,  sgd_lr)
	print("------------>Changed NOW:",K.get_value(model.optimizer.lr))


	print(model.summary())

	# for e,i in enumerate(XX):
	# 	e+=1
	# 	print("epoch..",e,"/",epochs)
		
	# 	seq_in,seq_out,_=overlap(i,4,t=8)
	# 	seq_in = seq_in.reshape((seq_in.shape[0], seq_in.shape[1], 64, 64, 1))
	# 	print(seq_in.shape)


	# 	if int(e) % 50 == 0:
	# 		print("------------>LR BEFORE:",K.get_value(model.optimizer.lr))
	# 		K.set_value(model.optimizer.lr,K.get_value(model.optimizer.lr) / 10)
	# 		print("------------>LR NOW:",K.get_value(model.optimizer.lr))

	history=model.fit(seq_in,seq_in,epochs=epochs,batch_size=batch_size,callbacks=[checkpointer])


	# history=model.fit_generator(generator(seq_in,batch_size,seq_in.shape[4]),steps_per_epoch=math.ceil(seq_in.shape[0] / batch_size),verbose=1,epochs=epochs,callbacks=[checkpointer])
	losss=[history.history['loss']]
	plot_history(losss,type)



def lstm_stateful(batch_size,timesteps,n_features):
	print("shape:---> ",timesteps,n_features)
	#dropout=dropout
	lstm_autoencoder = Sequential()
	# r=reg #works with 0.01
	# if units == 256:
	# 	r=0.1
	# # Encoderaaaaa
	# un=units*2
	lstm_autoencoder.add(LSTM(64, activation='relu',batch_input_shape=(batch_size,timesteps,n_features), stateful=True,return_sequences=True))
	# lstm_autoencoder.add(LSTM(un, activation='relu',return_sequences=True))
	lstm_autoencoder.add(LSTM(32, activation='linear',name="encoder",stateful=True))
	lstm_autoencoder.add(RepeatVector(timesteps))
	# Decoder ,activity_regularizer=regularizers.l1(r),activity_regularizer=regularizers.l1(r),activity_regularizer=regularizers.l2(r)
	lstm_autoencoder.add(LSTM(32, activation='relu',stateful=True,return_sequences=True))
	lstm_autoencoder.add(LSTM(64, activation='linear',stateful=True,return_sequences=True))
	# lstm_autoencoder.add(LSTM(32, activation='relu',input_shape=(timesteps, n_features), return_sequences=True))

	lstm_autoencoder.add(TimeDistributed(Dense(n_features)))

	lstm_autoencoder.summary()

	return lstm_autoencoder              

def test():
	visible  = Input(shape=(8, 64, 64, 3))
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(visible)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = LSTM(512, activation='relu',return_sequences=True)(x)
	# Bottleneck here!
	x = LSTM(32, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	pre_flat_shape[0] = 8
	x = Dense(np.product(pre_flat_shape), activation='relu')(x)
	x = Reshape(pre_flat_shape)(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(16, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	decoded = TimeDistributed(Conv2D(3, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)
	return model

def cnn_bilstm():
	'''
	Try to add LSTM in the Decoder
	'''
	visible  = Input(shape=(8, 64, 64, 1))
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(visible)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu',padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='valid'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(MaxPooling2D((2, 2), padding='valid'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
	res=x

	# x = TimeDistributed(MaxPooling2D((2, 2), padding='same'))(x)
	pre_flat_shape = x.shape[1:].as_list()
	x = TimeDistributed(Flatten())(x)
	x = Bidirectional(LSTM(16*16*1, name='Bottleneck' ,activation='linear'))(x)

	### x = LSTM(16*16*1, activation='linear',return_sequences=False)(x)

	# Bottleneck here!
	# x = LSTM(encoding, name='bottleneck', activation='linear')(x)
	# Start scaling back up
	# No frame stack for output
	
	# x = Dense(16*16*1, name='Bottleneck',activation='linear')(x)
	# x=Dropout(0.2)(x)
	x = Dense(512, activation='relu')(x)
	
	x = Reshape([8,8,8,1])(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x=Add()([x,res])
	x = TimeDistributed(Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(64, (3, 3),activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(UpSampling2D((2, 2)))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	x = TimeDistributed(Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
	decoded = TimeDistributed(Conv2D(1, (3, 3), activation='linear', padding='same'))(x)

	# this model maps an input to its reconstruction
	model=Model(inputs=visible,outputs=decoded)

	return model


if __name__ == "__main__":
   inputfile = ''
   level=''
   try:
      opts, args = getopt.getopt(sys.argv[1:],"hi:g:e:l:r:s:b:t:",["ifile=","level","epochs=","l=","r=","s=","batch=","type="])
   except getopt.GetoptError:
      print('test.py -i <inputfile>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print("test.py -i <inputfile>")
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-g", "--ifile"):
         level = arg #500,700,900 || 0 means all TBD
      elif opt in ("-e", "--epochs"):
      	 epochs=int(arg)
      elif opt in ("-l", "--l"):
      	 sgd_lr=float(arg)
      elif opt in ("-r","--r"):
      	 reg=float(arg)
      elif opt in ("-s","--s"):
      	 start=int(arg)
      elif opt in ("-b","--b"):
      	 batch_size=int(arg)
      elif opt in ("-t","--t"):
      	 typee=str(arg)
      	 # typee=str(typee)


   print('Input file is...', inputfile,'level...',level,' with epochs..',epochs,' and LR:',sgd_lr,'batch_size.: ',batch_size )

  
   LEVELS_STR = ['500', '700', '900']
   LEVELS = ['G5', 'G7', 'G9']

   print("level...",level)
   



   if level == 'all':
   		_LEVEL = LEVELS[LEVELS_STR.index('500')]
   		ght_500 = np.load(inputfile)[_LEVEL]
   		ght_500=ght_500.reshape(ght_500.shape[0],1,ght_500.shape[1])

   		_LEVEL = LEVELS[LEVELS_STR.index('700')]
   		ght_700 = np.load(inputfile)[_LEVEL]
   		ght_700=ght_700.reshape(ght_700.shape[0],1,ght_700.shape[1])

   		data=np.concatenate((ght_500,ght_700),axis=1)

   		_LEVEL = LEVELS[LEVELS_STR.index('900')]
   		ght_900 = np.load(inputfile)[_LEVEL]
   		ght_900=ght_900.reshape(ght_900.shape[0],1,ght_900.shape[1])

   		data=np.concatenate((data,ght_900),axis=1)

   		print(data.shape)


   else:

    	_LEVEL = LEVELS[LEVELS_STR.index(level)]
    	# Load GHT level
    	data = np.load(inputfile)[_LEVEL]
    	print(data.shape)

   

 
 
   if start == 1: 
   	#train NOISY mode with 1 GHT all noisy inputs
    X,train=load_dataset(data,epochs,start) ## X contains all noisy version of the dataset
    X_train1,train_y,_=overlap(X[0],4,t=8)   #700 GHT
   elif start==0:
   	##for 3 GHT
    # X,X1,X2=load_dataset(data,epochs,start)
    # X_train1,train_y,_=overlap(X,4,t=8) # 700 GHT 
    # X_train2,train_y2,_=overlap(X1,4,t=8) # 500 GHT
    # X_train3,train_y3,_=overlap(X2,4,t=8) # 900 GHT
    X=data.reshape(data.shape[0],data.shape[1]*data.shape[2])
   elif start==3:
    #only for conv_lstm return all in a list 
    X=load_dataset(data,epochs,start)
    print(X.shape)
    # X_train1,train_y,_=overlap(X,4,t=8)
   elif start==4:
    #plain no augmentation only the simple 700 GHT pressure
    X,train=load_dataset(data,epochs,start) ## all GHT
    X_train1,_,_=overlap(X,4,t=8) # 700 GHT 

   else:
    print('start is empty')
    sys.exit(2)





   # for units in units:
   for optimizer in optimizerss:
     print("Training UNITS: ",units,"optimizer:",optimizer,"batch_size",batch_size," lr",sgd_lr)
     # model_cnn_lstm_concat()
     # exit()
     for i in [256]:
     	model=test()
     	cnnlstm(X,batch_size,optimizer,sgd_lr,model)


     	###
    
     	# model1=model_cnnlstm4(i)
     	# cnnlstm(X,batch_size,optimizer,sgd_lr,model1)
     	# model=model_cnnlstm_transpose()
     	# # cnnlstm(X,batch_size,optimizer,sgd_lr,model)
     	# model1=model_cnn_lstm_concat()
     	# cnnlstm(X,batch_size,optimizer,sgd_lr,model1,lstm='yes')


     # mode_cnnlstm3()
#vfsdfsfdsfsdf
     # model=model_cnnlstm4()
     # cnnlstm(X,batch_size,optimizer,sgd_lr,model)

     ##STRIDEEEEE OF 1
    
     # cnnlstm1(X,batch_size,optimizer,sgd_lr)
    
   

     # super_stacked(X_train1.shape[1],X_train1.shape[2],optimizer,X_train1,X_train2,X_train3,epochs,batch_size,X,X1,X2,sgd_lr,'avg')
     # super_stacked(X_train1.shape[1],X_train1.shape[2],optimizer,X_train1,X_train2,X_train3,epochs,batch_size,X,X1,X2,sgd_lr,'concat')

     # conv_lstm(X,batch_size,optimizer,sgd_lr) ## maybe with adam and lr 0.1 ## 15 GIATI ETSI
     # model_conv_lstm(8)
   # autoencoder=lstm_stateful(batch_size,X_train1.shape[1],X_train1.shape[2])
   # train_ae(autoencoder,optimizer,loss,train,batch_size,epochs,reverse_reconstruction="false",flag=1,reset="true")

   # autoencoder=stacked_lstm_ae(X_train1.shape[1],X_train1.shape[2],'relu',units,optimizer,0.1)
   # train_ae(autoencoder,optimizer,loss,train,batch_size,epochs,sgd_lr,reverse_reconstruction="false",flag=1)


   
#         seq2seq_time_series_noisy(X_train1,X_train1.shape[1],X_train1.shape[2],optimizer,units,epochs,batch_size,train_y,X)
   #seq2seq_time_series_super(X_train1,X_train1.shape[1],X_train1.shape[2],optimizer,units,epochs,batch_size,X,X_train2,X_train3)

#         #seq2seq_rp(X_train1,X_train1.shape[1],X_train1.shape[2],optimizer,units,epochs,batch_size,train_y)
   

           


