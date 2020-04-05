"""
   CLASS INFO
   -------------------------------------------------------------------------------------------
    netCDF_subset class acts as an interface between any python script and any netCDF file.
    Enables data extraction, netCDF file reading and writing.
   -------------------------------------------------------------------------------------------
"""

import numpy as np
from netCDF4 import Dataset, num2date, date2num
from datetime import timedelta
import datetime


class netCDF_subset(object):
    _dataset = None  # initial netcdf dataset path
    _level_name = None
    _time_name = None
    _pressure_levels = None  # pressure level of interest
    _subset_variables = None  # variables of interest
    _time_unit = None
    _time_cal = None
    _ncar_lvl_names = None

    # Constructor
    def __init__(self, dataset, levels,
                 sub_vars, lvlname='level', timename='time',
                 time_unit='hours since 1900-01-01 00:00:00',
                 time_cal='gregorian', ncar_lvls=None):
        # Init original dataset
        self._dataset = Dataset(dataset, 'r')
        # Multiple levels
        self._pressure_levels = levels
        # Multiple vars
        self._subset_variables = sub_vars
        self._level_name = lvlname
        self._time_name = timename
        self._time_unit = time_unit
        self._time_cal = time_cal
        if ncar_lvls is None:
            self._ncar_lvl_names = np.array([0, 1, 2, 3, 5, 7, 10, 20, 30, 50,
                                             70, 100, 125, 150, 175, 200, 225,
                                             250, 300, 350, 400, 450, 500, 550,
                                             600, 650, 700, 750, 775, 800, 825,
                                             850, 875, 900, 925, 950, 975, 1000])
        try:
            self._sub_pos = self.lvl_pos()
        except:
            idx_list = []
            ncar = list(self._ncar_lvl_names)
            for lvl in self._pressure_levels:
                idx_list.append(ncar.index(lvl))
            self._sub_pos = idx_list

    # Find pressure level position in dataset
    def lvl_pos(self):
        idx_list = []
        arr = np.array(self._dataset.variables[self._level_name]).tolist()
        for lvl in self._pressure_levels:
            idx_list.append(arr.index(lvl))
        return idx_list

    # Retrieve variables for a specific level (defined in Class attributes)
    def extract_data(self):
        var_list = []
        for v in self._subset_variables:
            var_list.append(self._dataset.variables[v][:, self._sub_pos, :, :])
        return np.array(var_list)

    # Retrieve variables for a specific level and time (used in clusters to
    # file)
    def extract_timeslotdata(self, time_pos):
        var_list = []
        for v in self._subset_variables:
            var_list.append(self._dataset.variables[v][
                            time_pos, self._sub_pos, :, :])
        return np.array(var_list)

    def extract_piece(self, time_pos, lat_pos, lon_pos):
        var_list = []
        for v in self._subset_variables:
            var_list.append(self._dataset.variables[v][
                            time_pos, self._sub_pos, lat_pos, lon_pos])
        return np.array(var_list)

    def get_time_diagram(self, start_date, clut_list):
        if len(clut_list) != 1:
            raise ValueError(
                'List of clusters must contain only a single variable ' +
                'or a single list for multiple variables')
        times = self._dataset.variables[self._time_name][:]
        dist = []
        for c in clut_list[0]:
            sort_times = sorted(times[c])
            diffs = []
            for slots in sort_times:
                # Convert them to datetime objects in order to perform date
                # operations
                slots = num2date(slots, self._time_unit, self._time_cal)
                # Calculate the distances between start date and every other date
                # i.e: start_date = 1986-01-01_00:00
                # slot = 1986-01-02_08:00
                # diffs = 1 + 06/24 = 1.25
                diffs.append(float((slots - start_date).days) +
                             float(slots.hour) / 24)
            dist.append(diffs)
        return dist

    # Given a list of date time objects returns a list of indices of times that
    # their date distance is an hourslot (by default an hourslot is 6 hours)
    def find_hourslots(self, time_list, hourslot=6):
        idx_difs = []
        idx_dif = []
        for idx, t in enumerate(time_list):
            try:
                dif = (time_list[idx + 1] - time_list[idx]
                       ) == timedelta(hours=hourslot)
                # if the distance between those two is an hourslot then append
                # them
                if dif:
                    idx_dif.append(idx)
                # else if found a bigger distance then reset
                else:
                    idx_difs.append(idx_dif)
                    idx_dif = []
                    continue
            # if idx+1 out of bounds
            except:
                continue
        return idx_difs

    # Find the maximum continuous timeslot for every cluster in cluster_list
    def find_continuous_timeslots(self, clut_list, hourslot=6):
        if len(clut_list) != 1:
            raise ValueError(
                'List of clusters must contain only a single variable ' +
                'or a single list for multiple variables')
        times_list = []
        for pos, c in enumerate(clut_list):
            for nc in range(0, len(clut_list[0])):
                times = self._dataset.variables[self._time_name][c[nc]]
                times_list.append(
                    num2date(times, self._time_unit, self._time_cal))
        max_ret_list = []
        timestamps = []
        for c, time in enumerate(times_list):
            # Get hourslot indices
            idx_difs = self.find_hourslots(time, hourslot)
            len_list = []
            for idx in idx_difs:
                len_list.append(len(idx))
            # Find biggest continous hourslot containing list
            try:
                max_idx = max(len_list)
            except:
                max_ret_list.append([])
            else:
                pos_max = len_list.index(max_idx)
                start_idx = idx_difs[pos_max][0]
                end_idx = idx_difs[pos_max][len(idx_difs[pos_max]) - 1]
                timestamps.append([time[start_idx], time[end_idx]])
        # Find global indices instead of "local"(cluster indices)
        all_times = self._dataset.variables[self._time_name][:].tolist()
        for timestamp in timestamps:
            d2n = date2num(timestamp, self._time_unit, self._time_cal)
            start_idx = all_times.index(d2n[0])
            end_idx = all_times.index(d2n[1])
            max_ret_list.append((start_idx, end_idx))
        return max_ret_list

    def get_times(self):
        try:
            return num2date(self._dataset[self._time_name][:], self._time_unit, self._time_cal)
        except:
            return self._dataset[self._time_name][:]

    def find_time_slot(self, year, month, day, hour, min):
        dat = datetime.datetime(year, month, day, hour, min)
        return date2num(dat, self._time_unit, self._time_cal)

    def find_timeslot_idx(self,idx):
        times = self._dataset.variables[self._time_name][:]
        return num2date(times[idx],self._time_unit,self._time_name)

    # Create a season map and divide the dataset into seasons
    def get_seasons(self, times, season_ch):
        seasons_idx = []
        w_id = [12, 1, 2]
        sp_id = [3, 4, 5]
        su_id = [6, 7, 8]
        au_id = [9, 10, 11]
        for t in times:
            if t.month in w_id:
                seasons_idx.append('winter')
            elif t.month in sp_id:
                seasons_idx.append('spring')
            elif t.month in su_id:
                seasons_idx.append('summer')
            elif t.month in au_id:
                seasons_idx.append('autumn')
        winter_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'winter']
        spring_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'spring']
        summer_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'summer']
        autumn_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'autumn']
        if season_ch == 'winter':
            return winter_idx
        elif season_ch == 'summer':
            return summer_idx
        elif season_ch == 'spring':
            return spring_idx
        elif season_ch == 'autumn':
            return autumn_idx

    # Create season map and divide dataset into cold and hot seasons
    def get_biseasons(self, times, season_therm):
        seasons_idx = []
        cold_d = [9, 10, 11, 12, 1, 2]
        hot_d = [3, 4, 5, 6, 7, 8]
        for t in times:
            if t.month in cold_d:
                seasons_idx.append('cold')
            elif t.month in hot_d:
                seasons_idx.append('hot')
        cold_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'cold']
        hot_idx = [idx for idx, season in enumerate(
            seasons_idx) if season == 'hot']
        if season_therm == 'cold':
            return cold_idx
        elif season_therm == 'hot':
            return hot_idx

    # create dimensions used in write_tofile and write_timetofile
    def write_dimensions_to_file(self, dsout, time_pos=None):
        dim_vars = []
        for dname, dim in self._dataset.dimensions.iteritems():
            dim_vars.append(dname)
            if dname == self._level_name:
                dsout.createDimension(dname, len(
                    self._pressure_levels) if not dim.isunlimited() else None)
            elif dname == self._time_name:
                if not(time_pos is None):
                    dsout.createDimension(dname, len(
                        time_pos) if not dim.isunlimited() else None)
            else:
                dsout.createDimension(dname, len(dim)
                                      if not dim.isunlimited() else None)
        return dim_vars

    # write 4 dimension variables to netcdf output
    def write_var_to_file(self, dsout, v_name, varin, var_list, time_pos, pos_v, c_desc):
        outVar = dsout.createVariable(
            v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k)
                          for k in varin.ncattrs()})
        if c_desc:
            # make array flat
            reshape_dim = self._dataset[v].shape[
                2] * self._dataset[v].shape[3]
            mean = var_list[pos_v].reshape(
                len(time_pos), reshape_dim)
            # get array mean
            one = np.mean(np.matrix(mean), axis=0)
            # reshape into a single frame
            one = one.reshape(1, 1, self._dataset[v].shape[
                              2], self._dataset[v].shape[3])
            tt = var_list[pos_v][0, :]
            tt[:] = one[:]
            outVar[:] = tt
        else:
            outVar[:] = var_list[pos_v]

   # create variables used in write_tofile and write_timetofile
    def write_variables_to_file(self, dsout, dim_vars, var_list, time_pos=None, c_desc=False):
        for v_name, varin in self._dataset.variables.iteritems():
            if v_name in self._subset_variables:
                for pos_v, v in enumerate(self._subset_variables):
                    if v_name == v:
                        self.write_var_to_file(
                            dsout, v_name, varin, var_list, time_pos, pos_v,  c_desc)
            elif v_name in dim_vars:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                if v_name == self._level_name:
                    outVar[:] = self._pressure_levels
                elif v_name == self._time_name:
                    if not(time_pos is None):
                        outVar[:] = self._dataset.variables[
                            self._time_name][time_pos]
                else:
                    outVar[:] = varin[:]

    def write_gattrs_and_dims(self, dsout, time_pos, size=None, parts=None):
        for gattr in self._dataset.ncattrs():
            if gattr == 'SIMULATION_START_DATE':
                if isinstance(time_pos[0], np.ndarray):
                    gvalue = self._dataset.variables[
                        self._time_name][time_pos[0][0]]
                else:
                    gvalue = self._dataset.variables[
                        self._time_name][time_pos[0]]
                sim_date = gvalue.tostring()
                # for gv in gvalue:
                #     sim_date += gv
                print sim_date
                dsout.setncattr(gattr, sim_date)
            else:
                gvalue = self._dataset.getncattr(gattr)
                dsout.setncattr(gattr, gvalue)
        for dname, dim in self._dataset.dimensions.iteritems():
            if dname == self._time_name:
                if (size is None) and (parts is None):
                    dsout.createDimension(dname, len(
                        time_pos) if not dim.isunlimited() else None)
                else:
                    dsout.createDimension(dname, len(range(time_pos[0], time_pos[
                                          0] + (size / parts))) if not dim.isunlimited() else None)
            else:
                dsout.createDimension(dname, len(
                    dim) if not dim.isunlimited() else None)

    def write_var_to_file_mean(self, dsout, varin, v_name, time_pos, size, parts):
        # print varin.shape
        varin_flat = varin[time_pos, :].flatten()
        # print varin_flat.shape
        rang = range(0, size + parts, parts)
        temp_arr = np.array(0)
        for idx, i in enumerate(rang):
            try:
                start_idx = i * varin.shape[1] * \
                    varin.shape[2] * varin.shape[3]
                end_idx = rang[idx + 1] * varin.shape[1] * \
                    varin.shape[2] * varin.shape[3]
                one_frame = varin_flat[range(start_idx, end_idx)].reshape(
                    parts, varin.shape[1] * varin.shape[2] * varin.shape[3])
                varin_mean = np.mean(one_frame, axis=0)
                # print varin_mean.shape
                temp_arr = np.append(temp_arr, varin_mean)
            except:
                break
        temp_arr = np.delete(temp_arr, 0)
        temp_arr = temp_arr.reshape(
            size / parts, varin.shape[1], varin.shape[2], varin.shape[3])
        outVar = dsout.createVariable(
            v_name, varin.datatype, varin.dimensions)
        outVar.setncatts({k: varin.getncattr(k)
                          for k in varin.ncattrs()})
        # print outVar.shape
        outVar[:] = temp_arr

    def write_variables_to_file_mean(self, dsout, time_pos, size, parts, var_list):
        for v_name, varin in self._dataset.variables.iteritems():
            if v_name == self._time_name:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[
                    range(time_pos[0], time_pos[0] + (size / parts))]
            elif v_name == self._level_name:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[:]
            elif v_name in self._subset_variables and not(var_list is None):
                self.write_var_to_file(
                    dsout, v_name, varin, var_list, time_pos, self._subset_variables.index(v_name),  False)
            elif len(varin.shape) == 4:
                self.write_var_to_file_mean(
                    dsout, varin, v_name, time_pos, size, parts)
            else:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[
                    range(time_pos[0], time_pos[0] + (size / parts)), :]

    def write_variables_to_file_kmean(self, dsout, time_pos):
        time_range = []
        for t in range(0, len(time_pos)):
            time_range.append(time_pos[t][0])
        for v_name, varin in self._dataset.variables.iteritems():
            if v_name == self._time_name:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[time_range]
            elif v_name == self._level_name:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[:]
            else:
                varin_array = []
                for time in time_pos:
                    varin_array.append(varin[time])
                mean_array = []
                for var in varin_array:
                    mean_array.append(np.mean(var, axis=0))
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = mean_array[:]

    # Export results to file from attibute dataset
    def write_tofile(self, out_path, var_list=None):
        dsout = Dataset(out_path, 'w')
        dim_vars = []
        if var_list is None:
            var_list = self.extract_data(self.lvl_pos())
        dim_vars = self.write_dimensions_to_file(dsout, var_list)
        self.write_variables_to_file(dsout, dim_vars, var_list)
        dsout.close()

    # Export variables for specific lvl and time period
    def write_timetofile(self, out_path, lvl_pos, time_pos, c_desc=False):
        dsout = Dataset(out_path, 'w')
        dim_vars = []
        var_list = self.extract_timeslotdata(time_pos, lvl_pos)
        dim_vars = self.write_dimensions_to_file(dsout, time_pos)
        self.write_variables_to_file(
            dsout, dim_vars, var_list, time_pos, c_desc)
        dsout.close()

    # Export exact copy of netcdf with some variables/dimensions modified
    def exact_copy_file(self, out_path, time_pos):
        dsout = Dataset(out_path, 'w')
        self.write_gattrs_and_dims(dsout, time_pos)
        for v_name, varin in self._dataset.variables.iteritems():
            if v_name == self._level_name:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[:]
            elif v_name == self._time_name:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[time_pos]
            else:
                outVar = dsout.createVariable(
                    v_name, varin.datatype, varin.dimensions)
                outVar.setncatts({k: varin.getncattr(k)
                                  for k in varin.ncattrs()})
                outVar[:] = varin[time_pos, :]
        dsout.close()

    # Export exact copy of netcdf with some variables/dimensions modified
    def exact_copy_mean(self, out_path, time_pos, size, parts, var_list=None):
        dsout = Dataset(out_path, 'w')
        print size / parts
        self.write_gattrs_and_dims(dsout, time_pos, size, parts)
        self.write_variables_to_file_mean(
            dsout, time_pos, size, parts, var_list)
        dsout.close()

    def exact_copy_kmeans(self, out_path, time_pos):
        dsout = Dataset(out_path, 'w')
        self.write_gattrs_and_dims(dsout, time_pos)
        self.write_variables_to_file_kmean(dsout, time_pos)
        dsout.close()
