
def lstm_overlap_noisy(seq_in,n_in,d,optimizer,units,epochs,batch_size,seq_out,r,activation,train="true"):
##reconstruction_pred overlap window
        losss=[]
        t1=[]
        t2=[]
        historyy=[]
        dropout=0.2
        type="lstm_rp"+str(units)
        if r == 1:
        	type+="_reverse_reconstruction"
        if seq_in.shape[0]>=575:
        	type+="_1,5_day_overlap_"
        else:
        	type+="no_overlap"

        #plots=type+='.png'
        visible = Input(shape=(n_in,d))
        encoder = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.1))(visible)
        # define reconstruct decoder
        decoder1 = RepeatVector(n_in)(encoder)
        decoder1 = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.1), return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(d),name="reconstruction")(decoder1)
        # define predict decoder
        decoder2 = RepeatVector(n_in)(encoder)
        decoder2 = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.1), return_sequences=True)(decoder2)
        decoder2 = TimeDistributed(Dense(d),name="prediction")(decoder2)
        model = Model(inputs=visible, outputs=[decoder1, decoder2])
        print(model.summary())
        print(type,"train rec_pred_ae for:",epochs,"optimizer: ",optimizer)
        model.compile(optimizer=optimizer, loss=loss)
        plot_model(model, to_file='type.png', show_shapes=True, show_layer_names=True)
        #early=EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None,$
        if train == "true":
	        filepath='tmp/' + type + '.hdf5'
	        log_dir='./logs_' + type
	        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
	        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
	       
        	if r==1:
        		history=model.fit(seq_in, [np.flip(seq_in),seq_out],epochs=epochs,batch_size=batch_size,shuffle=True,callbacks=[tbCallBack,checkpointer,TerminateOnNaN()])
        		losss.append(history.history['loss'][0])
        		t1.append(history.history['reconstruction_loss'][0])
        		t2.append(history.history['prediction_loss'][0])
        	else:
        		history=model.fit(seq_in, [seq_in,seq_out],epochs=epochs,batch_size=batch_size,shuffle=True,callbacks=[tbCallBack,checkpointer,TerminateOnNaN()])
        		losss.append(history.history['loss'][0])
        		t1.append(history.history['reconstruction_loss'][0])
        		t2.append(history.history['prediction_loss'][0])
 
	        #historyy.append(losss)
	        historyy.append(t1)
	        historyy.append(t2)

	        
	        if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
	        	return

	        plot_history(historyy,type)

	        yhat=model.predict(seq_in)
	        yhat=yhat[0] ## only the reconstruction 
	        if r ==1:
			
	                yhat=np.flip(yhat)

	        #print(yhat[1])
	        #print(seq_in[1])
	        
	        print("shape---->",yhat.shape,seq_in.shape)
		
	        print_figs(seq_in,yhat,3,type)

	        ##keep the encoder for clustering
	        model = Model(inputs=model.inputs, outputs=model.layers[1].output)
	        yhat = model.predict(seq_in)

	        clustering(yhat,type)
        

def lstm_overlap(seq_in,n_in,d,optimizer,units,epochs,batch_size,seq_out,r,activation,X,train="true"):
        ##reconstruction_pred overlap window
        losss=[]
        t1=[]
        t2=[]
        historyy=[]
        dropout=0.2
        type="lstm_rp"+str(units)
        if r == 1:
        	type+="_reverse_reconstruction"
        if seq_in.shape[0]>=575:
        	type+="_1,5_day_overlap_"
        else:
        	type+="no_overlap"

        #plots=type+='.png'
        visible = Input(shape=(n_in,d))
        encoder = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.1))(visible)
        # define reconstruct decoder
        decoder1 = RepeatVector(n_in)(encoder)
        decoder1 = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.1), return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(d),name="reconstruction")(decoder1)
        # define predict decoder
        decoder2 = RepeatVector(n_in)(encoder)
        decoder2 = LSTM(units, activation=activation,dropout=dropout,activity_regularizer=regularizers.l2(0.1), return_sequences=True)(decoder2)
        decoder2 = TimeDistributed(Dense(d),name="prediction")(decoder2)
        model = Model(inputs=visible, outputs=[decoder1, decoder2])
        print(model.summary())
        print(type,"train rec_pred_ae for:",epochs,"optimizer: ",optimizer)
        model.compile(optimizer=optimizer, loss=loss)
        plot_model(model, to_file='type.png', show_shapes=True, show_layer_names=True)
        #early=EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None,$
        if train == "true":
	        filepath='tmp/' + type + '.hdf5'
	        log_dir='./logs_' + type
	        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
	        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
	        for n,i in enumerate(X):
	        	print("epochs..",n+1,"/",epochs)
	        	seq_in,seq_out,test=overlap(i,8,t=8)
	        	if r==1:
	        		history=model.fit(seq_in, [np.flip(seq_in),seq_out],epochs=1,batch_size=batch_size,shuffle=True,callbacks=[tbCallBack,checkpointer,TerminateOnNaN()])
	        		losss.append(history.history['loss'][0])
	        		t1.append(history.history['reconstruction_loss'][0])
	        		t2.append(history.history['prediction_loss'][0])
	        	else:
	        		history=model.fit(seq_in, [seq_in,seq_out],epochs=1,batch_size=batch_size,shuffle=True,callbacks=[tbCallBack,checkpointer,TerminateOnNaN()])
	        		losss.append(history.history['loss'][0])
	        		t1.append(history.history['reconstruction_loss'][0])
	        		t2.append(history.history['prediction_loss'][0])
 
	        #historyy.append(losss)
	        historyy.append(t1)
	        historyy.append(t2)

	        
	        if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
	        	return

	        plot_history(historyy,type)

	        yhat=model.predict(seq_in)
	        yhat=yhat[0] ## only the reconstruction 
	        if r ==1:
			
	                yhat=np.flip(yhat)

	        #print(yhat[1])
	        #print(seq_in[1])
	        
	        print("shape---->",yhat.shape,seq_in.shape)
		
	        print_figs(seq_in,yhat,3,type)

	        ##keep the encoder for clustering
	        model = Model(inputs=model.inputs, outputs=model.layers[1].output)
	        yhat = model.predict(seq_in)

	        clustering(yhat,type)
        
     

def lstm_rp(seq_in,n_in,d,optimizer,units,epochs,batch_size):
	#lstm_rp_simple no winodws
	seq_out = seq_in[:, 1:, :] ## this could be y 
	n_out = n_in - 1
	type+="lstm_reconstruct_predict_simple"
	visible = Input(shape=(n_in,d))
	encoder = LSTM(units, activation='tanh')(visible)
	# define reconstruct decoder
	decoder1 = RepeatVector(n_in)(encoder)
	decoder1 = LSTM(units, activation='tanh', return_sequences=True)(decoder1)
	decoder1 = TimeDistributed(Dense(d))(decoder1)
	# define predict decoder
	decoder2 = RepeatVector(n_out)(encoder)
	decoder2 = LSTM(units, activation='tanh', return_sequences=True)(decoder2)
	decoder2 = TimeDistributed(Dense(d))(decoder2)
	model = Model(inputs=visible, outputs=[decoder1, decoder2])
	print(model.summary())
	#model.compile(optimizer=optimizer, loss='mse')
	#plot_model(model, to_file='lstm_rc_plot.png', show_shapes=True, show_layer_names=True)
	#model.fit(seq_in, [seq_in,seq_out], epochs=300,batch_size=32) #default 32
	# demonstrate prediction
	#yhat = model.predict(seq_in)
	#print(yhat.shape)
	
	print("train rec_pred_ae for:",epochs,"optimizer: ",optimizer,"loss: ", loss)
	model.compile(optimizer=optimizer, loss=loss)
	#early=EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None,$
	filepath='tmp/' + type + '.hdf5'
	tbCallBack = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
	checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
	history=model.fit(seq_in, [seq_in,seq_out],
                  epochs=epochs,
                  batch_size=batch_size,shuffle=False,callbacks=[checkpointer,TerminateOnNaN(),tbCallBack])
	

	yhat=model.predict(seq_in)
        #print(yhat[1])
        #print(seq_in[1])
	yhat=yhat[0]
	print("---->",yhat.shape,seq_in.shape)
	
def decode_sequence(input_seq):
		# from our previous model - mapping encoder sequence to state vectors
	encoder_model = Model(encoder_inputs, encoder_states)

	# A modified version of the decoding stage that takes in predicted target inputs
	# and encoded state vectors, returning predicted target outputs and decoder state vectors.
	# We need to hang onto these state vectors to run the next step of the inference loop.
	decoder_state_input_h = Input(shape=(latent_dim,))
	decoder_state_input_c = Input(shape=(latent_dim,))
	decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

	decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
	decoder_states = [state_h, state_c]

	decoder_outputs = decoder_dense(decoder_outputs)
	decoder_model = Model([decoder_inputs] + decoder_states_inputs,
	                  [decoder_outputs] + decoder_states)


	# Encode the input as state vectors.
	states_value = encoder_model.predict(input_seq)

	# Generate empty target sequence of length 1.
	target_seq = np.zeros((1, 1, 1))

	# Populate the first target sequence with end of encoding series pageviews
	target_seq[0, 0, 0] = input_seq[0, -1, 0]

	# Sampling loop for a batch of sequences - we will fill decoded_seq with predictions
	# (to simplify, here we assume a batch of size 1).

	decoded_seq = np.zeros((1,pred_steps,4096))

	for i in range(pred_steps):
	    
	    output, h, c = decoder_model.predict([target_seq] + states_value)
	    
	    decoded_seq[0,i,0] = output[0,0,0]

	    # Update the target sequence (of length 1).
	    target_seq = np.zeros((1, 1, 1))
	    target_seq[0, 0, 0] = output[0,0,0]

	    # Update states
	    states_value = [h, c]
	return decoded_seq


def seq2seq_time_series_super(seq_in,n_in,d,optimizer,units,epochs,batch_size,X,seq_in_y,seq_in_z):
  latent_dim = units # LSTM hidden units
  dropout = .20

  # Define an input series and encode it with an LSTM. 
  encoder_inputs = Input(shape=(None,d)) 
  encoder_inputs_y = Input(shape=(None,d)) 
  encoder_inputs_z = Input(shape=(None,d)) 

  encoder_y = LSTM(latent_dim, return_state=True)
  encoder_z = LSTM(latent_dim, return_state=True)
  encoder = LSTM(latent_dim, return_state=True)


  encoder_outputs, state_h, state_c = encoder(encoder_inputs)
  encoder_outputs_y, state_h_y, state_c_y = encoder(encoder_inputs)
  encoder_outputs_z, state_h_z, state_c_z = encoder(encoder_inputs)
  state_h=Average(name='avg')([state_h,state_h_y,state_h_z])
  state_c=Average(name='avg1')([state_c,state_c_y,state_c_z])

  # We discard `encoder_outputs` and only keep the final states. These represent the "context"
  # vector that we use as the basis for decoding.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  # This is where teacher forcing inputs are fed in.
  decoder_inputs = Input(shape=(None, d))
  decoder_inputs_y = Input(shape=(None, d))
  decoder_inputs_z = Input(shape=(None, d))
  # We set up our decoder using `encoder_states` as initial state.  
  # We return full output sequences and return internal states as well. 
  # We don't use the return states in the training model, but we will use them in inference.
  decoder_lstm = LSTM(latent_dim,  return_sequences=True, return_state=True)
  decoder_lstm_y = LSTM(latent_dim,  return_sequences=True, return_state=True)
  decoder_lstm_z = LSTM(latent_dim,  return_sequences=True, return_state=True)

  decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)
  decoder_outputs_y, _, _ = decoder_lstm_y(decoder_inputs,initial_state=encoder_states)
  decoder_outputs_z, _, _ = decoder_lstm_z(decoder_inputs,initial_state=encoder_states)



  decoder_dense = Dense(d)# The dimensionality of the output at each time step. In this case a 1D signal.
  decoder_dense_y = Dense(d)
  decoder_dense_z = Dense(d)
  # There is no reason for the input sequence to be of same dimension as the ouput sequence.
  # For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.
  decoder_outputs = decoder_dense(decoder_outputs)
  decoder_outputs_y = decoder_dense_y(decoder_outputs_y)
  decoder_outputs_z = decoder_dense_z(decoder_outputs_z)
  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs,encoder_inputs_y,encoder_inputs_z, decoder_inputs,decoder_inputs_y,decoder_inputs_z], [decoder_outputs,decoder_outputs_y,decoder_outputs_z])
  type="super_seq2seq_"+str(units)+str(optimizer)+".png"
  #plot_model(model, to_file=type, show_shapes=True, show_layer_names=True)

  print(model.summary())

  model.compile(optimizer=optimizer, loss=loss)
  # log_dir='./logs_'+"seq2seq_time_series" +str(units)
  # tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
  ##why y ? -> y is lagged in 1 day..
  filepath='tmp/' + type + '.hdf5'
  checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
  history = model.fit([seq_in,seq_in_y,seq_in_z,np.flip(seq_in),np.flip(seq_in_y),np.flip(seq_in_z)],[np.flip(seq_in),np.flip(seq_in_y),np.flip(seq_in_z)],batch_size=batch_size,epochs=epochs,callbacks=[TerminateOnNaN(),checkpointer])


  plot_history(list(history.history.values()),"super_seq2seq_time_series_"+str(units)+str(optimizer))

  # yhat=model.predict(seq_in)
  # yhat=np.flip(yhat)

  #print(yhat[1])
  #print(seq_in[1])


  # print_figs(seq_in,yhat,3,type)

  ##keep the encoder for clustering
  encoder_model = Model(encoder_inputs, encoder_states)
  yhat = encoder_model.predict(seq_in)
  print()
  yhat=np.array(yhat)
  print("shape---->",yhat.shape)
  yhat=yhat[0]
  clustering(yhat,type)
  print("YHAT-------->",yhat)
  type="seq2seq_"+str(units)+str(optimizer)
  savp=type+"_representation.npy"
  np.save(savp,yhat)
  ##------------------------#
  win=4
  t=8
  kmeans = KMeans(n_clusters=15,n_jobs=-1,n_init=20)
  km=kmeans.fit(yhat)
  pred_kmeans = kmeans.predict(yhat)
  z = TSNE(n_components=2, verbose=2).fit_transform(yhat)
  plot_tsne(z,pred_kmeans)
  closest(km,yhat,X)

def seq2seq_time_series(seq_in,n_in,d,optimizer,units,epochs,batch_size,X):
  latent_dim = units # LSTM hidden units
  dropout = .20

  # Define an input series and encode it with an LSTM. 
  encoder_inputs = Input(shape=(None,d)) 
  # encoder_inputs_y = Input(shape=(None,d)) 
  # encoder_inputs_z = Input(shape=(None,d)) 

  encoder = LSTM(latent_dim, return_state=True)
  encoder_outputs, state_h, state_c = encoder(encoder_inputs)

  # We discard `encoder_outputs` and only keep the final states. These represent the "context"
  # vector that we use as the basis for decoding.
  encoder_states = [state_h, state_c]

  # Set up the decoder, using `encoder_states` as initial state.
  # This is where teacher forcing inputs are fed in.
  decoder_inputs = Input(shape=(None, d))

  # We set up our decoder using `encoder_states` as initial state.  
  # We return full output sequences and return internal states as well. 
  # We don't use the return states in the training model, but we will use them in inference.
  decoder_lstm = LSTM(latent_dim,  return_sequences=True, return_state=True)
  decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

  decoder_dense = Dense(d)# The dimensionality of the output at each time step. In this case a 1D signal.
  # There is no reason for the input sequence to be of same dimension as the ouput sequence.
  # For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.
  decoder_outputs = decoder_dense(decoder_outputs)

  # Define the model that will turn
  # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  type="seq2seq_"+str(units)+str(optimizer)+".png"
  #plot_model(model, to_file=type, show_shapes=True, show_layer_names=True)

  print(model.summary())

  model.compile(optimizer=optimizer, loss=loss)
  filepath='tmp/' + type + '.hdf5'
  checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')

  # log_dir='./logs_'+"seq2seq_time_series" +str(units)
  # tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
  ##why y ? -> y is lagged in 1 day..
  history = model.fit([seq_in, np.flip(seq_in)],np.flip(seq_in),batch_size=batch_size,epochs=epochs,callbacks=[TerminateOnNaN(),checkpointer])


  plot_history(list(history.history.values()),"seq2seq_time_series_"+str(units)+str(optimizer))

  # yhat=model.predict(seq_in)
  # yhat=np.flip(yhat)

  #print(yhat[1])
  #print(seq_in[1])


  # print_figs(seq_in,yhat,3,type)

  ##keep the encoder for clustering
  encoder_model = Model(encoder_inputs, encoder_states)
  yhat = encoder_model.predict(seq_in)
  print()
  yhat=np.array(yhat)
  print("shape---->",yhat.shape)
  yhat=yhat[0]
  clustering(yhat,type)
  print("YHAT-------->",yhat)
  type="seq2seq_"+str(units)+str(optimizer)
  savp=type+"_representation.npy"
  np.save(savp,yhat)
  ##------------------------#
  win=4
  t=8
  kmeans = KMeans(n_clusters=15,n_jobs=-1,n_init=20)
  km=kmeans.fit(yhat)
  pred_kmeans = kmeans.predict(yhat)
  z = TSNE(n_components=2, verbose=2).fit_transform(yhat)
  plot_tsne(z,pred_kmeans)
  closest(km,yhat,X)


        




def seq2seq_rp(seq_in,n_in,d,optimizer,units,epochs,batch_size,seq_out):
	latent_dim = units # LSTM hidden units
	dropout = .20

	# Define an input series and encode it with an LSTM. 
	encoder_inputs = Input(shape=(None,d)) 
	encoder = LSTM(latent_dim, dropout=dropout, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	# We discard `encoder_outputs` and only keep the final states. These represent the "context"
	# vector that we use as the basis for decoding.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	# This is where teacher forcing inputs are fed in.
	decoder_inputs = Input(shape=(None, d))

	# We set up our decoder using `encoder_states` as initial state.  
	# We return full output sequences and return internal states as well. 
	# We don't use the return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

	decoder_dense = Dense(d)# The dimensionality of the output at each time step. In this case a 1D signal.
	# There is no reason for the input sequence to be of same dimension as the ouput sequence.
	# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.
	decoder_outputs = decoder_dense(decoder_outputs)


	decoder_inputs_pred = Input(shape=(None, d))

	# We set up our decoder using `encoder_states` as initial state.  
	# We return full output sequences and return internal states as well. 
	# We don't use the return states in the training model, but we will use them in inference.
	decoder_lstm_pred = LSTM(latent_dim, dropout=dropout, return_sequences=True, return_state=True)
	decoder_outputs_pred, _, _ = decoder_lstm_pred(decoder_inputs_pred,initial_state=encoder_states)

	decoder_dense_pred = Dense(d)# The dimensionality of the output at each time step. In this case a 1D signal.
	# There is no reason for the input sequence to be of same dimension as the ouput sequence.
	# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.
	decoder_outputs_pred = decoder_dense_pred(decoder_outputs_pred)


	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs,decoder_inputs,decoder_inputs_pred], [decoder_outputs,decoder_outputs_pred])
	type="seq2seq_"+str(units)+str(optimizer)+".png"
	plot_model(model, to_file=type, show_shapes=True, show_layer_names=True)

	print(model.summary())

	model.compile(optimizer=optimizer, loss=loss)
	log_dir='./logs_'+"seq2seq_time_series" +str(units)
	tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
	##why y ? -> y is lagged in 1 day..
	history = model.fit([seq_in, np.flip(seq_in),np.flip(seq_in)],[np.flip(seq_in),np.flip(seq_in)],batch_size=batch_size,epochs=epochs,callbacks=[tbCallBack,TerminateOnNaN()])


	plot_history(list(history.history.values()),"seq2seq_time_series_"++str(units)+str(optimizer))


def seq2seq_time_series_noisy(seq_in,n_in,d,optimizer,units,epochs,batch_size,seq_out,X):
	latent_dim = units # LSTM hidden units
	dropout = .20

	# Define an input series and encode it with an LSTM. 
	encoder_inputs = Input(shape=(None,d)) 
	encoder = LSTM(latent_dim, dropout=dropout,activity_regularizer=regularizers.l2(0.01), return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)

	# We discard `encoder_outputs` and only keep the final states. These represent the "context"
	# vector that we use as the basis for decoding.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	# This is where teacher forcing inputs are fed in.
	decoder_inputs = Input(shape=(None, d))

	# We set up our decoder using `encoder_states` as initial state.  
	# We return full output sequences and return internal states as well. 
	# We don't use the return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, dropout=dropout,activity_regularizer=regularizers.l2(0.01), return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states)

	decoder_dense = Dense(d)# The dimensionality of the output at each time step. In this case a 1D signal.
	# There is no reason for the input sequence to be of same dimension as the ouput sequence.
	# For instance, using 3 input signals: consumer confidence, inflation and house prices to predict the future house prices.
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	type="seq2seq_"+str(units)+str(optimizer)+".png"
	plot_model(model, to_file=type, show_shapes=True, show_layer_names=True)

	print(model.summary())
	losss=[]
	historyy=[]

	model.compile(optimizer=optimizer, loss=loss)
	log_dir='./logs_'+"seq2seq_time_series" +str(units)
	tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
	early=EarlyStopping(monitor='loss', min_delta=0, patience=10, verbose=0, mode='auto')

	##why y ? -> y is lagged in 1 day..
	for n,i in enumerate(X):
		print("epochs..",n+1,"/",epochs)
		seq_in,seq_out,test=overlap(i,8,t=8)
		history = model.fit([seq_in, np.flip(seq_in)],np.flip(seq_out),batch_size=batch_size,epochs=1,callbacks=[tbCallBack,TerminateOnNaN(),early])
		losss.append(history.history['loss'][0])
	historyy.append(losss)


	plot_history(historyy,"seq2seq_time_series_"+str(units)+str(optimizer))


def lstm_rp_stateful(seq_in,n_in,d,optimizer,units,epochs,batch_size,seq_out,r,activation,train="true"):
		##simple_stateful
        historyy=[]
        losss=[]
        t1=[]
        t2=[]
        type="lstm_rp_stateful_"+ str(units)
        if r==1:
        	type+="_reverse_reconstruction"
        # seq_out = seq_in[:, 1:, :]
        #n_out = n_in -1
        visible = Input(batch_shape=(26,n_in,d))
        encoder = LSTM(units, activation=activation,stateful=True)(visible)
        # define reconstruct decoder
        decoder1 = RepeatVector(n_in)(encoder)
        decoder1 = LSTM(units, activation=activation, stateful=True,return_sequences=True)(decoder1)
        decoder1 = TimeDistributed(Dense(d),name='reconstruction')(decoder1)
        # define predict decoder
        decoder2 = RepeatVector(n_in)(encoder)
        decoder2 = LSTM(units, activation=activation, stateful=True,return_sequences=True)(decoder2)
        decoder2 = TimeDistributed(Dense(d),name='prediction')(decoder2)
        model = Model(inputs=visible, outputs=[decoder1, decoder2])
        print(model.summary())

        if train=="true":
        #plot_model(model, to_file='lstm_rc_plot.png', show_shapes=True, show_layer_names=True)
  
	        print(type,"train rec_pred_ae for:",epochs,"optimizer: ",optimizer,"loss: ", loss)
	        model.compile(optimizer=optimizer, loss=loss)
	        
	        filepath='tmp/' + type + '.hdf5'
	        log_dir='./logs_' + type
	        tbCallBack = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)
	        checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')
	        ##not shuffle in stateful
	        if r==1:
	        	for i in range(epochs):
	        		print("epochs...",i,"/",epochs)
	        		history=model.fit(seq_in, [np.flip(seq_in),seq_out],epochs=1,batch_size=26,shuffle=False,callbacks=[tbCallBack,checkpointer,TerminateOnNaN()])

	        		losss.append(history.history['loss'][0])
	        		t1.append(history.history['reconstruction_loss'][0])
	        		t2.append(history.history['prediction_loss'][0])
	        		if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
	        			return

	        		model.reset_states()


	        	historyy.append(losss)
	        	historyy.append(t1)
	        	historyy.append(t2)

	        	# historyy=historyy[::-1]
	              
	        else:
	            for i in range(epochs):
	            	print("epochs....",i,"/",epochs)
	            	history=model.fit(seq_in, [seq_in,seq_out],epochs=1,batch_size=26,shuffle=False,callbacks=[tbCallBack,checkpointer,TerminateOnNaN()])
	            	losss.append(history.history['loss'][0])
	            	t1.append(history.history['reconstruction_loss'][0])
	            	t2.append(history.history['prediction_loss'][0])
	            	model.reset_states()
	            	if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
	            		return

	            historyy.append(losss)
	            historyy.append(t1)
	            historyy.append(t2)
	            print(history.history.keys())
	            # historyy=historyy[::-1]

	       	

	        #full encoder-decode to compare the reconstruction with the initial input
	        print(history.history['loss'][0])
	        if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
	        	return
	        yhat=model.predict(seq_in,batch_size=26)
	        yhat=yhat[0] ## only the reconstruction 

	        if r ==1:
			
	                yhat=np.flip(yhat)

	        print("---->",yhat.shape,seq_in.shape)
	        plot_history(historyy,type)

	        print_figs(seq_in,yhat,3,type)

		        ##keep the encoder for clustering
	        model = Model(inputs=model.inputs, outputs=model.layers[1].output)

	        yhat = model.predict(seq_in,batch_size=26)

	        clustering(yhat,type)



def train_X_ae(autoencoder,optimizer,loss,x_train,x_test,batch_size,epochs,reset="false",reverse_reconstruction="false",noise="false"):
  #train ae
  type="lstm_plain_"+str(units)+str(optimizer)
  print(type," train ae for:",epochs,"optimizer: ",optimizer,"loss: ", loss)
  autoencoder.compile(optimizer=optimizer, loss=loss)
  #early=EarlyStopping(monitor='loss', min_delta=0, patience=50, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
  if noise=="true":
    type+="_noise"
  if reset=="true":
    type+="_stateful"
    historyy=[]
    losss=[]
    t1=[]

  if reverse_reconstruction == "true":
    ##reverse reconstruction
    x_test=np.flip(x_test) ##::-1
    type+="_reverse_reconstruction"

  filepath='tmp/' + type + '.hdf5'
  log_dir='./logs_'+type
  tbCallback = TensorBoard(log_dir=log_dir, histogram_freq=0, write_graph=True, write_images=True)

  checkpointer = ModelCheckpoint(filepath=filepath, verbose=1,monitor='loss',save_best_only=True,mode='min')

  if reset=="true":
    
    for i in range(epochs):
        print("epochs...",i,"/",epochs)
        history=autoencoder.fit(x_train, x_test,epochs=1,batch_size=batch_size,shuffle=False,callbacks=[tbCallback,checkpointer,TerminateOnNaN()])
        autoencoder.reset_states()
        if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
           return
        losss.append(history.history['loss'][0])

    historyy.append(losss)
    # historyy=historyy[::-1]

  else:
    history=autoencoder.fit(x_train, x_test,epochs=epochs,batch_size=batch_size,shuffle=True,callbacks=[tbCallback,checkpointer,TerminateOnNaN()])
  print(history.history['loss'][0])
  if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
    return
  if reset=="true":
    plot_history(historyy,type)
  else:

   plot_history(list(history.history.values()),type)

  seq_in=x_train

  if reset!="true":
       yhat=autoencoder.predict(seq_in)
  else:
        yhat=autoencoder.predict(seq_in,batch_size=batch_size)
  if reverse_reconstruction == "true":
        yhat=np.flip(yhat)

  print("---->",yhat.shape,seq_in.shape)
  print_figs(seq_in,yhat,3,type)
    
    #clustering the encodings only
  model = Model(inputs=autoencoder.inputs, outputs=autoencoder.get_layer("encoder").output)
  if reset=="true":
        yhat = model.predict(seq_in,batch_size=batch_size)
  else:
        yhat = model.predict(seq_in)
    
  clustering(yhat,type)


##vae



def create_lstm_vae(timesteps, 
    input_dim, 
    batch_size, 
    intermediate_dim, 
    latent_dim,
    optimizer,
    X,
    epsilon_std=1.):

    x = Input(shape=(timesteps, input_dim))

    # LSTM encoding
    h = LSTM(intermediate_dim,dropout=0.2,activity_regularizer=regularizers.l2(0.01))(x)

    # VAE Z layer
    z_mean = Dense(latent_dim,activity_regularizer=regularizers.l2(0.01))(h)
    z_log_sigma = Dense(latent_dim,activity_regularizer=regularizers.l2(0.01))(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                  mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon

    # note that "output_shape" isn't necessary with the TensorFlow backend
    # so you could write `Lambda(sampling)([z_mean, z_log_sigma])`
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, dropout=0.2,activity_regularizer=regularizers.l2(0.01),return_sequences=True)
    decoder_mean = LSTM(input_dim,dropout=0.2,activity_regularizer=regularizers.l2(0.01),return_sequences=True)

    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)

    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)

    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)

    # generator, from latent space to reconstructed inputs
    decoder_input = Input(shape=(latent_dim,))

    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)

    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = losses.mean_absolute_error(x, x_decoded_mean)
        kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss

    vae.compile(optimizer=optimizer, loss=vae_loss)

    print(vae.summary())
    for e,i in enumerate(X):
      print("epoch..",e,"/",epochs)
      print(i.shape)
      X_train,x_test,test=overlap(i,8,t=8)
      # if e!=0 and e==10 and e!=epochs:
      #   print("------------>LR BEFORE:",K.get_value(autoencoder.optimizer.lr))
      #   K.set_value(optimizer.lr,  K.get_value(optimizer.lr) / 10)
      #   print("------------>LR NOW:",K.get_value(autoencoder.optimizer.lr))

      # if e!=0 and e==120 and e!=epochs:
      #   print("------------>LR BEFORE:",K.get_value(autoencoder.optimizer.lr))
      #   K.set_value(optimizer.lr,  K.get_value(optimizer.lr) / 10)
      #   print("------------>LR NOW:",K.get_value(autoencoder.optimizer.lr))
        
     # model.lr.set_value(.02)
     
      history=vae.fit(X_train, X_train,epochs=1,batch_size=batch_size,shuffle=True)

      losss.append(history.history['loss'][0])

    historyy.append(losss)
  

    if np.isnan(history.history['loss'][0]) or np.isinf(history.history['loss'][0]):
      return
    if reset=="true":
      plot_history(historyy,type)
    else:
      plot_history(historyy,type)

    X_train,x_test,test=overlap(np.array(X[0]),8,t=8)
    seq_in=X_train

   
    yhat=vae.predict(seq_in)

    print("---->",yhat.shape,seq_in.shape)
    print_figs(seq_in,yhat,3,type)
    
    #clustering the encodings only
  #print(autoencoder.layers)

 
        
    yhat = encoder.predict(seq_in)

    print("previous Mean..",seq_in.mean())
    print("YHAT-------->",yhat)
    savp=type+"_representation.npy"
    np.save(savp,yhat)
    clustering(yhat,type)