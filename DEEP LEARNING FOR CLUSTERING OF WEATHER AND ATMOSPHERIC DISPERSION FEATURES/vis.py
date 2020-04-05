import numpy as np
from sklearn.manifold import TSNE
import getopt
import sys
import math
import pydot
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from tensorflow.python.keras.utils import plot_model
import matplotlib

import collections 

def plot_history(history,type):
	#print(list(history.history.values()))
	#print(history.history.keys())
	for i in range(len(history)):
	    	# print(history[i])
	    	plt.plot(history[i])
	    	# print(list(history.history.values())[i])
	    
	plt.ylabel('Mean Absolute Error Loss')
	if len(history) > 1:
		plt.legend(['rec_loss','pred_loss'])
	else:
		plt.legend(['loss'])
	plt.xlabel('Epoch')
	#plt.ylabel('Mean Absolute Error Loss')
	title='Loss Over Time' + type
	plt.title(title)
	#plt.legend(['Train','Valid'])
	plt.show()
	type="history_"+type+".png"
	plt.savefig(type)
	
	plt.clf()
	plt.cla()
	plt.close()
	###
	    





def print_figs(seq_in,yhat,pics,name):
	##overlap windows may look different from the non-overlapped.
	sseq_in=seq_in
	syhat=yhat
	seq_in1=seq_in
	yhat1=yhat
	seq_in = seq_in[0].reshape(len(seq_in[0]), 64, 64)
	yhat = yhat[0].reshape(len(yhat[0]), 64, 64)
	print(yhat.shape,seq_in.shape)
	fig = plt.figure()
	for i in range(1,pics+1):
		ax = fig.add_subplot(2, pics, i)
		#pixels = image.reshape((x, y))
		ax.matshow(seq_in[0], cmap=matplotlib.cm.binary)
		ax2 = fig.add_subplot(2, pics, i+pics)
		#pixels2 = image2.reshape((x, y))
		ax2.matshow(yhat[0], cmap=matplotlib.cm.binary)

	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.show()
	type=name
	t="figs_"+type+".png"
	plt.savefig(t)

	plt.clf()
	plt.cla()
	plt.close()

	## PRINT 2 FIGS

	seq_in = seq_in1[5].reshape(len(seq_in1[5]), 64, 64)
	yhat = yhat1[5].reshape(len(yhat1[5]), 64, 64)
	print(yhat.shape,seq_in.shape)
	fig = plt.figure()
	for i in range(1,pics+1):
		ax = fig.add_subplot(2, pics, i)
		#pixels = image.reshape((x, y))
		ax.matshow(seq_in[5], cmap=matplotlib.cm.binary)
		ax2 = fig.add_subplot(2, pics, i+pics)
		#pixels2 = image2.reshape((x, y))
		ax2.matshow(yhat[5], cmap=matplotlib.cm.binary)

	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.show()
	type=name
	t="figs2_"+type+".png"
	plt.savefig(t)
	
	plt.clf()
	plt.cla()
	plt.close()

	seq_in = seq_in1[6].reshape(len(seq_in1[6]), 64, 64)
	yhat = yhat1[6].reshape(len(yhat1[6]), 64, 64)
	print(yhat.shape,seq_in.shape)
	fig = plt.figure()
	for i in range(1,pics+1):
		ax = fig.add_subplot(2, pics, i)
		#pixels = image.reshape((x, y))
		ax.matshow(seq_in[6], cmap=matplotlib.cm.binary)
		ax2 = fig.add_subplot(2, pics, i+pics)
		#pixels2 = image2.reshape((x, y))
		ax2.matshow(yhat[6], cmap=matplotlib.cm.binary)

	plt.xticks(np.array([]))
	plt.yticks(np.array([]))
	plt.show()
	type=name
	t="figs3_"+type+".png"
	plt.savefig(t)
	print("prin--->",syhat.shape,sseq_in.shape)

	yhat = syhat.reshape((syhat.shape[0]*syhat.shape[1], syhat.shape[2]))
	seq_in = sseq_in.reshape((sseq_in.shape[0]*sseq_in.shape[1], sseq_in.shape[2]))
	n = 10  
	plt.figure(figsize=(20, 4))
	for i in range(n):
	    # display original
	    ax = plt.subplot(2, n, i + 1)
	    plt.imshow(seq_in[i].reshape(64, 64))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)

	    # display reconstruction
	    ax = plt.subplot(2, n, i + 1 + n)
	    plt.imshow(yhat[i].reshape(64, 64))
	    plt.gray()
	    ax.get_xaxis().set_visible(False)
	    ax.get_yaxis().set_visible(False)
	plt.show()
	type=name
	t="figs4_"+type+".png"
	plt.savefig(t)
	plt.clf()
	plt.cla()
	plt.close()

def plot_cluster_accuracy(scores_silhouette,name,type):
	
	#plt.plot(scores_calinski)
	plt.plot(scores_silhouette)
	plt.ylabel('Acc')
	plt.legend(['silhouette_score'])
	plt.xlabel('Clusters +3 ')
	title='Cluster Accuracy' + type 
	plt.title(title)
	plt.show()
	#plt.legend(['Train','Valid'])
	type="cluster_"+name+type+".png"
	plt.savefig(type)
	
	plt.clf()
	plt.cla()
	plt.close()

from datetime import datetime

def plot_tsne(xy, colors=None,latent=None, alpha=1, figsize=(6,6), s=10, cmap='viridis'):
    # from sklearn.manifold import TSNE
    # xy=np.load(xy) 
    # xy=TSNE(n_components=2, verbose=2, perplexity=40, n_iter=300).fit_transform(xy)
    # if os.path.exists("tsne.png"):
    #     os.remove("tsne.png")
    # else:
    #     print("The .png file does not exist")
    plt.figure(figsize=figsize, facecolor='white')
    plt.margins(0)
    plt.axis('on')
    fig = plt.scatter(xy[:,0], xy[:,1],
                c=colors, # set colors of markers
                cmap=cmap, # set color map of markers
                marker=',', # use smallest available marker (square)
                s=s, # set marker size. single pixel is 0.5 on retina, 1.0 otherwise
                lw=0	, # don't use edges
                edgecolor='') # don't use edges
    # remove all axes and whitespace / borders
    fig.axes.get_xaxis().set_visible(True)
    fig.axes.get_yaxis().set_visible(True)
    plt.grid(True)

    import random

    plt.scatter(xy[106][0],xy[106][1],marker="x",color='r',s=100)

    plt.scatter(xy[10050][0],xy[205][1],marker="o",color='r',s=100)
    #plt.scatter(xy[50][0],xy[50][1],marker="o",color='r',s=100)

    plt.scatter(xy[85][0],xy[1005][1],marker="s",color='r',s=100)

    plt.scatter(xy[3010][0],xy[3010][1],marker="+",color='r',s=100)

    plt.scatter(xy[8000][0],xy[400][1],marker="d",color='r',s=100)
    #plt.scatter(xy[300][0],xy[300][1],marker="o",color='r',s=100)
    # y=random.randint(0,len(xy)-1)
    # plt.scatter(xy[y][0],xy[y][1],marker="d",color='r',s=100)

    # plt.scatter(xy[20][0],xy[20][1],marker="d",color='r',s=100)
    if latent is not None:
	    print("distances...")
	    from scipy.spatial import distance
	    # print("distances... from 32 space visualized in 2d space...","with circle: ",distance.euclidean(latent[6],latent[1050]),"with square: ",distance.euclidean(latent[6],latent[85]),"with plus: ",distance.euclidean(latent[6],latent[1]),"with diamond..",distance.euclidean(latent[6],latent[500]))
	    print("distances... from 32 space visualized in 2d space...","with circle: ",distance.euclidean(latent[106],latent[10050]),"with square: ",distance.euclidean(latent[106],latent[85]),"with plus: ",distance.euclidean(latent[106],latent[3010]),"with diamond..",distance.euclidean(latent[106],latent[8000]))

    pl=str(datetime.now())+"tsne.png"
    plt.savefig(pl)
    print("fig ploted")

def plot_class_space_tSNE(z, labels, out=None, distinct=False):
    z = TSNE(n_components=2, verbose=2).fit_transform(z)
    #np.save('_tsne.npy', z)
    fig = plt.figure()
    if distinct:
        color = plt.get_cmap("viridis")
        color = color(np.linspace(0, 1, 15))
    else:
        color = cm.rainbow(np.linspace(0, 1, 15))
    for l, c in zip(range(15), color):
        ix = np.where(labels == l)
        #  plt.scatter(z[ix, 0], z[ix, 1], c=c, label=l, s=8, linewidth=0)
        plt.scatter(z[ix, 0], z[ix, 1], c=c, lsabel=l, s=8, linewidth=0)
    # plt.legend()
    if out:
        fig.savefig(out + ".pdf", dpi=1000)
    else:
        plt.show()


def plot_2(yhat,X,name):
        print("---->",yhat.shape,seq_in.shape)
        seq_in = seq_in[1].reshape(len(seq_in[1]), 64, 64)
        yhat = yhat[1].reshape(len(yhat[1]), 64, 64)
        print(yhat.shape,X_train.shape)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        #pixels = image.reshape((x, y))
        ax.matshow(seq_in[0], cmap=matplotlib.cm.binary)
        ax2 = fig.add_subplot(1, 2, 2)
        #pixels2 = image2.reshape((x, y))
        ax2.matshow(yhat[0], cmap=matplotlib.cm.binary)
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.plot()
        plt.show()
        t=name+".png"
        plt.savefig(t)

def validation():
        history = [x for x in X_train]
        # walk-forward validation over each week
        predictions = list()
        for i in range(len(X_test)):
                # predict the week
                yhat_sequence = forecast(model, history, 28)
                # store the predictions
                predictions.append(yhat_sequence)
                # get real observation and add to history for predicting the next week
                history.append(X_test[i, :])
        # evaluate predictions days for each week
        predictions = array(predictions)
        print(predictions.shape)
    