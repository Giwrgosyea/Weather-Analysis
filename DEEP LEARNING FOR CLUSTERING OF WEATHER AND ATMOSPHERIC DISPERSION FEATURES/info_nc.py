import netCDF4 as netcdf4
import sys
import os

# a=netcfd4.dataset(....nc)
# a=['Times'][:].toString()



if __name__ == "__main__":
	for l in sys.argv[1:]:
		if l.endswith('.nc'):
			print("file:",l)
			info=netcdf4.Dataset('../wrfhy/wrfvol/nc/'+str(l))
			print(info['Times'][:])