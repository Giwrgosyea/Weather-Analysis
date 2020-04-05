import numpy as np
import os 
from vis import plot_tsne
from sklearn.manifold import TSNE
import getopt
import sys
from sklearn.cluster import KMeans
import numpy as np
import math
import datetime 
import netCDF4 as netcdf4
import shutil
import datetime

inputfile=''
PREFIX = "GHT700_"
if __name__ == "__main__":
  try:
    opts, args = getopt.getopt(sys.argv[1:],"hi:",["ifile="])
  except getopt.GetoptError:
    print('test.py -i <inputfile>')
    sys.exit(2)
  for opt, arg in opts:
    if opt == '-h':
       print("test.py -i <inputfile>")
       sys.exit()
    elif opt in ("-i", "--ifile"):
       inputfile = arg

  print('Input file is...', inputfile) 
  dataset=np.load(inputfile)
  print("dataset..",dataset.shape)
  dataset=dataset[:54663,1,:] ##700 pressure
  print("700 pressure dataset..",dataset.shape)
  sys.path.insert(1,'../final_eval/')
  os.chdir("../final_eval")
  from Dataset_transformations import Dataset_transformations
  from Clustering import Clustering
  from netcdf_subset import netCDF_subset
  import dataset_utils as utils
  print(os.getcwd())

  sys.path.insert(1,'../wrfhy/wrfvol/')
  os.chdir("../wrfhy/wrfvol/")
  export_template = netCDF_subset('40years.nc', [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
  
  sys.path.insert(1,'../../final_eval/')
  os.chdir("../../final_eval/")

  ds = Dataset_transformations(dataset, 1000, dataset.shape)
  print ds._items.shape
  # times = export_template.get_times()
  # nvarin = []
  # for var in export_template.get_times():
  #     if var[0] != 'masked':
  #       str=''
  #       for v in var:
  #           str += v
  #       # print(str)
  #       nvarin.append(str)
  # times = []
  # for var in nvarin:
  #     under_split = var.split('_')
  #     date_split = under_split[0].split('-')
  #     time_split = under_split[1].split(':')
  #     date_object = datetime.datetime(int(date_split[0]), int(date_split[1]), int(date_split[2]), int(time_split[0]), int(time_split[1]))
  #     times.append(date_object)
  # print times[0:10]

  # print(len(times))

  # print ds._items.shape
  clust_obj = Clustering(ds,n_clusters=15,n_init=100,features_first=False)
  clust_obj.batch_means()
  # clust_obj.create_density_descriptors(8,times) # 8 6hour snapshot = 2 days
  clust_obj.create_km2_descriptors(12)

  sys.path.insert(1,'../wrfhy/wrfvol/')
  os.chdir("../wrfhy/wrfvol/")
  
  export_template = netCDF_subset(
      '40years.nc', [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
  export_template._time_name = 'Times'

  sys.path.insert(1,'../../final_eval/')
  os.chdir("../../final_eval/")
  a=os.getcwd()

  import dataset_utils as utils
  utils.export_descriptor_kmeans(str(a),export_template,clust_obj)
  clust_obj.save(PREFIX+'_mult_dense.zip')


