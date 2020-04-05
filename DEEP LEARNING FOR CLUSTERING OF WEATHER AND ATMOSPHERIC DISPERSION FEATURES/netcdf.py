import numpy as np 
import getopt
import sys
import os
import netCDF4 as netcdf4

#"../../weather2/2019-07-23 09:14:29.394383times_pos.npy"

inputfile='' ##should be time_pos
cluster=''

def desc_date(clust_obj,nc_subset,time_idx):
    desc_date = []
    for pos,i in enumerate(time_idx):
        gvalue = nc_subset._dataset.variables[nc_subset._time_name][i[0]]
        sim_date = ""
        for gv in gvalue:
            sim_date += gv
        desc_date.append(sim_date)
    clust_obj._desc_date = desc_date

if __name__ == "__main__":
	try:
		opts, args = getopt.getopt(sys.argv[1:],"hi:c:",["ifile=","cluster"])
	except getopt.GetoptError:
		print('test.py -i <inputfile>')
		sys.exit(2)
	for opt, arg in opts:
		if opt == '-h':
		   print("test.py -i <inputfile>") #time pos file ls
		   sys.exit()
		elif opt in ("-i", "--ifile"):
		   inputfile = arg
		# elif opt in ("-c", "--cluster"):
		# 	cluster=  arg



	a=np.load(inputfile)   

	w=list() 
	for i in range(0,len(a),8): 
		t=list() 
		for j in range(i,i+8,1): 
			t.append(int(a[j])) 
		w.append(t)

	print(w)	
	
	import sys
	sys.path.insert(1,'../wrfhy/wrfvol')
	from netcdf_subset import netCDF_subset as ns
	os.chdir("../wrfhy/wrfvol")
	print(os.getcwd())
	data = ns('40years.nc', [500,700,900], ['GHT'])
	data._time_name = 'Times'
	sys.path.insert(1,'../../final_eval')
	from dataset_utils import load_single as ls
	os.chdir("../../final_eval")
	print(os.path.dirname(os.path.realpath(__file__)))
	cluster_obj=ls('cluster.zip')
	time_idx=a
	desc_date(cluster_obj,data,w)
	cluster_obj.save('cluster.zip')
	sys.path.insert(1,'../wrfhy/wrfvol')
	os.chdir("../wrfhy/wrfvol")
	import shutil

	for e,i in enumerate(w):
		name = 'exp_cluster_'+str(e)+'.nc'
		data.exact_copy_file(name, i)
		info=netcdf4.Dataset(name)
		for i in info['Times'][:]:
				print(i.tostring())
		
		# if "06:00:00" in info['Times'][:][0].tostring():
		# 	print("failed")
		# 	os.remove(name)
		# 	i = [j-1 for j in i]
		# 	print(i)
		# 	name='exp_cluster_'+str(e)+"_edit_"+'.nc'
		# 	data.exact_copy_file(name, i)
		# 	info=netcdf4.Dataset(name)
		# 	for i in info['Times'][:]:
		# 		print(i.tostring())
		# else:
		# 	for i in info['Times'][:]:
		# 		print(i.tostring())


		print('moving:',name)
		shutil.move(name,'nc/')

	
	