import numpy as np
import matplotlib as mpl
import os 
if os.environ.get('DISPLAY','') == '':
        print('no display found. Using non-interactive Agg backend')
        mpl.use('Agg')
import matplotlib.pyplot as plt

from vis import plot_tsne
from sklearn.manifold import TSNE
import getopt
import sys
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics import pairwise_distances
import numpy as np
import math
import datetime 
import netCDF4 as netcdf4
import shutil
import datetime


def find_cluster_items(pred_kmeans,win,t,clusters,years):
  print("searching items for cluster... :",clusters)
  c=dict()
  for i in clusters:
    c[i]=[]
  

  print(set(pred_kmeans))
  #gather cluster items
  for e,i in enumerate(pred_kmeans):
    if i in c.keys():
      c[i].append(e)
  #cluster[1]->[1,2,4,6,...] => position in order to look back
  #print(c)

  sys.path.insert(1,'../wrfhy/wrfvol')
  from netcdf_subset import netCDF_subset as ns
  import os
  os.chdir("../wrfhy/wrfvol")
  print(os.getcwd())
  data = ns('40years.nc', [500,700,900], ['GHT'])
  data._time_name = 'Times'
  
  ##this should be for years years[2000]->[2days,2days,2days...]
  print('--->',c)
  clusteridx=dict()

  # if len(years)==0:

  #   for j in c:
  #     # years=[]
  #     for i in c[j]:
  #       times=[]
  #       #print("starting..",i)
  #       #print('2 days slot..',(i*win),':',(i*win)+t)
  #       for k in range((i*win),(i*win)+t,1) :
  #         #print("putting:",k)
  #         times.append(int(k))

  #       name = 'exp_cluster_'+str(i)+'.nc'
  #       data.exact_copy_file(name, times)
  #       info=netcdf4.Dataset(name)
  #       index=[]
  #       year=0
  #       for m in info['Times'][:]:
  #         tt=m.tostring()
  #         temp=tt.split('_')
  #         year=temp[0][:4]
  #         # print(year)
  #         if year not in years:
  #           years.append(year)
  #       os.remove(name)
      # print('cluster ',j,'years',years)

      # clusteridx[j]=years

  # np.save('clusteridx',clusteridx)
  # print(clusteridx)


  final_y=dict()
  for i in years:
    final_y[i]=list()

  print(final_y)
  # print(c)
  for j in c:
    cidx=[]
    for i in c[j]:
      times=[]
      #print("starting..",i)
      #print('2 days slot..',(i*win),':',(i*win)+t)
      for k in range((i*win),(i*win)+t,1) :
        #print("putting:",k)
        times.append(int(k))

      name = 'exp_cluster_'+str(i)+'.nc'
      data.exact_copy_file(name, times)
      info=netcdf4.Dataset(name)
      index=[]
      year=0
      for m in info['Times'][:]:
        tt=m.tostring()
        temp=tt.split('_')
        year=temp[0][:4]
        #print(year)
        if year in years:
          #print(year)
          index.append(m.tostring())

      #print(index)
      #print(year)
      if len(years)!=0:
        if year in years:
          #print(year)
          final_y[year].append(index[0])
          cidx.append(year)
          os.remove(name)
        else:
          os.remove(name)
      else:
        # if year in final_y.keys():
        #   final_y[year].append(index[0])
        # else:
        #   final_y[year]=index[0]
        # os.remove(name)
        print('insert years..')
        return
    print("----------------------------------------")
    clusteridx[j]=set(cidx)
  
  # print(final_y)
  # print(len(final_y))
  # if len(final_y) ==1:
  #   print(final_y)
  #   print('empty..')
  #   return
  np.save("clusteridx.npy",clusteridx)
  plots=dict()
  for i in final_y:
    plots[i]=[]
  

  print(final_y)
  print(plots)
  

  for i in final_y:
    seen=dict()
    for j in final_y[i]:
      
      temp=j.split('_')
      year=temp[0][:4]


      if years==None:
        print("insert years...")
        print(temp)
        seen[temp[0]]=[year]
      else:
        if year in years:

          # t_date=temp[0].split(year)
          # date=t_date[1]
          #print(temp)
          print(temp)
          seen[temp[0]]=[year]

      print(seen)

    plots[i].append(seen)

  print('PLOTS')
  print(plots)
  np.save('plots.npy',plots)
  import os
  
  sys.path.insert(1,'../../weather2')
  os.chdir("../../weather2")
  print(os.getcwd())

  sums=[0 for x in range(372)]
  xxx=0
  for k in plots:
    pplots=dict()
    if len(plots[k][0]) == 0:
       for i in range(1,13,1):
          for j in range(1,32,1):
            if i<10 and j <10:
                add=str(k)+'-0'+str(i)+'-0'+str(j)
                pplots[add]=0
            elif i < 10 and j >=10:
                add=str(k)+'-0'+str(i)+'-'+str(j)
                pplots[add]=0
            elif i >=10 and j < 10:
                add=str(k)+'-'+str(i)+'-0'+str(j)
                if add not in pplots.keys():
                  pplots[add]=0
            else:
                add=str(k)+'-'+str(i)+'-'+str(j)
                pplots[add]=0

    else:  

      for j in plots[k][0]:

        pplots[j]=len(plots[k][0][j])

      for i in range(1,13,1):
        for j in range(1,32,1):
          if i<10 and j <10:
              add=pplots.keys()[0].split('-')[0]+'-0'+str(i)+'-0'+str(j)
              if add not in pplots.keys():
                pplots[add]=0
          elif i < 10 and j >=10:
              add=pplots.keys()[0].split('-')[0]+'-0'+str(i)+'-'+str(j)
              if add not in pplots.keys():
                pplots[add]=0
          elif i >=10 and j < 10:
              add=pplots.keys()[0].split('-')[0]+'-'+str(i)+'-0'+str(j)
              if add not in pplots.keys():
                pplots[add]=0
          else:
              add=pplots.keys()[0].split('-')[0]+'-'+str(i)+'-'+str(j)
              if add not in pplots.keys():
                pplots[add]=0

  

    sorted_pplots=[]
    ## sort dict keys
    # print(pplots.keys())
    print(pplots)
    for a in sorted(pplots.keys()):
      sorted_pplots.append(pplots[a]) ## counts

    print(len(sorted_pplots))
    sums=[sum(x) for x in zip(sums,sorted_pplots)]
    xxx=sorted(pplots.keys())

    import datetime
    from matplotlib.pyplot import figure
    figure(num=None, figsize=(70, 30), dpi=80, facecolor='w', edgecolor='k')
    plt.ylabel('Score')
    plt.legend(['Times occured'])
    plt.xlabel('2days Dates')
    plt.bar(range(len(sorted(pplots.keys()))), sorted_pplots,align='center')
    plt.xticks(range(len(sorted(pplots.keys()))), sorted(pplots.keys()),rotation=90)
    pl=str(clusters[0])+str(datetime.datetime.now())+"_"+str(k)+"_bar.png"
    plt.savefig(pl)
    print("fig ploted")
  

  # from matplotlib.pyplot import figure
  # figure(num=None, figsize=(70, 30), dpi=80, facecolor='w', edgecolor='k')
  # plt.ylabel('Score')
  # plt.legend(['Times occured'])
  # plt.xlabel('2days Dates')

  # # We can set the number of bins with the `bins` kwarg
  # plt.bar(range(len(xxx)), sums,align='center')
  # plt.xticks(range(len(xxx)), xxx,rotation=90)

  # pl=str(clusters[0])+str(datetime.datetime.now())+"_"+str(k)+"_bar_all_years.png"
  # plt.savefig(pl)
  # print("fig ploted")





def closest(km, latent, num_min, win=4, t=8, save=True):
  # closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, latent)
  closest = pairwise_distances(km.cluster_centers_, latent)

  ret = []
  # Retrieve top 8 closest
  for i in range(closest.shape[0]):
    current_cent = closest[i, :]
    dist_sort = sorted(current_cent)[:num_min]
    idc = [list(current_cent).index(j) for j in dist_sort]
    ret.append(idc)
  
  #  weather=list()
  times_pos=list()
  for i in ret:
    current_time_pos = []
    for idx in i:
        lst_idx = []
        print("starting..",idx)
        print('adding..',(idx*win),':',(idx*win)+t)
        #  weather.append(dataset[(i*win):(i*win)+t])
        for j in range((idx*win),(idx*win)+t,1):
          print("putting:",j)
          lst_idx.append(j)
        current_time_pos.append(lst_idx)
    times_pos.append(current_time_pos)

  print("times_pos:",times_pos)

  if save == True:
    #  weather=np.array(weather)
    #  np.save('weather_'+str(datetime.datetime.now())+'_.npy',weather)
    times_pos=np.array(times_pos)
    print(times_pos.shape)
    np.save('times_pos_'+str(datetime.datetime.now())+'_.npy',times_pos)

    print("saved...")

  return times_pos




inputfile = ''
lat=''
# win=4
win=4 ## to allaksa evala overlap
t=8
save=False
z=None
c=0

if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:t:w:l:s:z:c:",["ifile=","time","win","lat","save","tsne","cluster"])
  except getopt.GetoptError:
    print('test.py -i <inputfile>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print("test.py -i <inputfile>")
       sys.exit()
    elif opt in ("-i", "--ifile"):
       inputfile = arg
    elif opt in ("-t", "--time"):
       t = arg
    elif opt in ("-w", "--win"):
       w = arg 
    elif opt in ("-l", "--lat"):
       lat=arg
    elif opt in ("-s", "--save"):
       save=True
    elif opt in ("-z", "--tsne"):
       z=arg
    elif opt in ("-c", "--cluster"):
       c=1

  print('Input file is...', inputfile) 
  print('latent rep file is...', lat)
  latent=np.load(lat)
  
  latent=np.append(latent,(latent[-1].reshape(1,latent.shape[1])),axis=0)
  # LEVELS_STR = ['500', '700', '900']
  # LEVELS = ['G5', 'G7', 'G9']
  # # _LEVEL = LEVELS[LEVELS_STR.index('700')]
  # # dataset = np.load(inputfile)[_LEVEL]

  print("latent:",latent.shape)
  sys.path.insert(1,'../final_eval/')
  from Dataset_transformations import Dataset_transformations
  from Clustering import Clustering
  import dataset_utils 




  ds = Dataset_transformations(latent, 1000, latent.shape)
  # kmeans = KMeans(n_clusters=15,n_jobs=-1,n_init=20)
  # km=kmeans.fit(latent)
  # pred_kmeans = kmeans.predict(latent)
  clust_obj = Clustering(ds,n_clusters=15,n_init=100,features_first=False)
  kmeans=clust_obj.kmeans()

  if save==True:
    clust_obj.save('cluster_mult_dense.zip')
  pred_kmeans = clust_obj._link.predict(latent)
  print(set(pred_kmeans))
  km=clust_obj._link
  if z==None:
  	print('running tsne..')
  	z = TSNE(n_components=2, verbose=2).fit_transform(latent)
  	
  # print("---------->",z)
  else:
  	print('loading tsne...')
  	z=np.load(z)

  sys.path.insert(1,'../weather2/')
  if save == True:
    zz=np.array(z)
    np.save('z_'+str(datetime.datetime.now())+'.npy',zz)
  # print("lens",len(z),len(latent))
  ## import clusters
  # print(pred_kmeans[0],z[0],len(pred_kmeans),len(z)
  # if c ==1 :
    ##one cluster each time ..
  # plot_tsne(z,pred_kmeans,latent)
  find_cluster_items(pred_kmeans,win,t,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14],['2017','1998','2005','2006','1986','1979','2008','1984','2011','2018','2015','1990','1989','2003','1999','1992','2013','2001','1985','2014','1994','2004','1981','2000','1980','2009','2010','2007','1995','1991','2016','1997','1996','1988','1993','2002','2012','1982','1983','1987'])

  # # 
  # t=closest(km,latent,dataset,save)