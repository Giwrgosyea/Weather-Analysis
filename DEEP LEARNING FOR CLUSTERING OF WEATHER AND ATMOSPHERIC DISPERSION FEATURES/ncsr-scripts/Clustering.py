"""
   CLASS INFO
   -------------------------------------------------------------------------------------------
   Clustering class contains every method that is necessary for creating,exporting and
   evaluating clusters. Inherits the Dataset/Dataset_transformations which makes it indepedent
   of data origin and structure.
   -------------------------------------------------------------------------------------------
"""

import numpy as np
from Dataset import Dataset
from sklearn.cluster import KMeans, MiniBatchKMeans, AgglomerativeClustering
from sklearn import metrics
import dataset_utils as utils
# import oct2py
from sklearn.neighbors.kde import KernelDensity
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm
import sys
from scipy.signal import argrelextrema
import operator


def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

class Clustering(Dataset):

    def __init__(self, dataset, n_clusters, n_init,
                 max_iter=300, features_first=False, similarities=None):
        super(Clustering, self).__init__(
            dataset.get_items(), dataset.get_items_iterator(), dataset.get_similarities())
        # Flag that indicates if the dataset shape is in (samples,features) order
        # or (features,samples)
        self._features_first = features_first
        data = self.get_items()
        # if (features,samples) then transpose
        if self._features_first:
            self._items = np.transpose(self.get_items())
        self._n_clusters = n_clusters # cluster number
        self._n_init = n_init # number of times that k-means will be repeated
        self._max_iter = max_iter # maximum number of iterations of the k-means algorithm for a single run
        self._clustering_dist = None # sample distirbution in clusters
        self._labels = None # cluster assignment array
        self._centroids = None # cluster centroids
        self._index_list = None  # result of k-means ( list of lists of int )
        self._link = None # sklearn KMeans object

    def kmeans(self):
        data = self.get_items()
        # print data.shape
        self._link = KMeans(n_clusters=self._n_clusters, n_init=self._n_init,
                            max_iter=self._max_iter, n_jobs=-1).fit(data)
        self._labels = self._link.labels_
        self._centroids = self._link.cluster_centers_
        self.get_clut_list(self._labels)

    def batch_kmeans(self, max_no_imprv):
        self._max_no_imprv = max_no_imprv
        data = self.get_items()
        init_size = self.get_items_iterator()
        self._link = MiniBatchKMeans(n_clusters=self._n_clusters, init_size=init_size, n_init=self._n_init,
                                     max_iter=self._max_iter, max_no_improvement=self._max_no_imprv, random_state=123).fit(data)
        self._labels = self._link.labels_
        self._centroids = self._link.cluster_centers_
        self.get_clut_list(self._labels)

    def hierachical(self, affinity, linkage):
        self._affinity = affinity
        self._linkage = linkage
        data = self.get_items()
        self._link = AgglomerativeClustering(
            n_clusters=self._n_clusters, affinity=self._affinity, linkage=self._linkage).fit(data)
        self._labels = self._link.labels_
        self._leaves = self._link.leaves_
        self.get_clut_list(self._labels)

    def create_km2_descriptors(self, frames):
        """
        This function outputs cluster descriptors using KMeans clustering twice.
        Once the KMeans clustering has been completed and led to splitting the samples
        in clusters, we proceed in repeating the k-means clustering individually in each
        cluster using k=<frames>, where frames is a number indicating the exported netcdf
        time range in slots of 6 hours (e.g 3 days = 72 hours = 12 frames of 6 hour slots).
        Each frame being the outcome of the second clustering is lated averaged and becomes
        a frame of the exported netcdf file descriptor.
        """
        data = self.get_items()
        # Get initial k-means result
        clut_list = self._index_list[0]
        c_desc = []
        # for each cluster (c contains the indices of initial samples)
        for c in clut_list:
            # Repeat k-means for each individual cluster
            cluster_data = data[c]
            kj = KMeans(n_clusters=frames, n_init=self._n_init,
                        max_iter=self._max_iter, n_jobs=-1).fit(cluster_data).labels_
            avg = []
            # Convert local indices to "global indices" so we are able to backtrace
            # which data need to be averaged in order to become a frame.
            for j in range(0, frames):
                idx = [idx for idx, frame in enumerate(
                    kj) if frame == j]
                avg.append(c[idx])
            c_desc.append(avg)
        self._descriptors = c_desc

    def create_density_descriptors(self, frames, times, lmax_limit=5):
        """
        Create max-density cluster descriptors.
        Params:
        times: list of datetime objects
        frames: int - the number of 6hr-snapshots required
        """
        bwdth = 32*2
        snap_duration_hrs = 6
        # Transpose all dates to a fixed hypothetical year and calculate hour offsets
        refdate = datetime(2020, 1, 1)
        times_f = np.array([ int((d.replace(year=2020) -
                           refdate).total_seconds() / 3600.0) for d in times ])
        data = self.get_items() # samples, features
        clusters = self._index_list[0]
        c_descriptors = []  # the cluster descriptors

        frames_filled = frames_total = 0
        # print len(times), len(times_f), len(data), np.max(times_f)
        for c in clusters:

            indexes = np.array([x for x in c])
            cdata = data[indexes]
            ctimes_f = times_f[indexes]
            X_plot = np.linspace(0, times_f[-1], times_f[-1])[:, np.newaxis]
            kde = KernelDensity(kernel='gaussian', bandwidth=bwdth).fit(
                ctimes_f[:, np.newaxis])
            dens = np.exp(kde.score_samples(X_plot))

            # Find the local maxima
            max_density_indexes = argrelextrema(dens, np.greater)[0]
            print(max_density_indexes)
            local_max = []
            for md in max_density_indexes:
                local_max.append((md, dens[md]))
            local_max.sort(key=operator.itemgetter(1), reverse=True)
            if len(local_max) > lmax_limit:
                local_max = local_max[0:lmax_limit]
            # for oi in local_max:
            #     plt.axvline(x=oi[0], color='r', linestyle='--')
            # print local_max

            # Iterate through all local maxes selected above
            c_descs = []  # All the descriptors of the current cluster
            for lm in local_max:
                c_desc = [] # the descriptor of the current local maximum
                max_den_i = lm[0]
                # find positions of interest
                cent_pos = max_den_i - (max_den_i % snap_duration_hrs)
                if max_den_i % snap_duration_hrs > snap_duration_hrs / 2:
                    cent_pos += snap_duration_hrs
                start_time_offset = cent_pos - (frames / 2) * snap_duration_hrs  ##
                end_time_offset = start_time_offset + frames * snap_duration_hrs ##

                pos = []  # list of time offsets
                for k in range(frames):
                    frames_total += 1
                    pos.append(start_time_offset + k * snap_duration_hrs)
                    cindices = np.where(np.in1d(ctimes_f, pos[k]))[0] # indices in cluster data list where the offsets occur - it may be []

                    if len(cindices) > 0:
                        gindices = indexes[cindices]
                        # print times[indexes[cindices]]  # checking real times
                        c_desc.append(np.array(gindices))
                        # c_desc.append(np.mean(cdata[cindices], 0))           # ***
                    else:
                        c_desc.append(None)

                # Deal with None by duplicating neighbouring snapshots (shouldn't occur often...)
                for k in range(frames):
                    if c_desc[k] is None and k > 0:
                        c_desc[k] = c_desc[k-1]
                        if c_desc[k] is not None: frames_filled += 1
                for k in reversed(range(frames)):
                    if c_desc[k] is None and k < frames - 1:
                        c_desc[k] = c_desc[k+1]
                        if c_desc[k] is not None: frames_filled += 1

                # for displaying data - need to uncommend *** above
                # from disputil import display_array
                # for tmp in c_desc:
                #     img = np.array(tmp).reshape((64,64))
                #     display_array(img)

                # for displaying the descriptor ranges in the hypothetical year
                # plt.axvline(x=start_time_offset, color='r', linestyle='--')
                # plt.axvline(x=end_time_offset, color='r', linestyle='--')
                # plt.plot(X_plot, dens, 'k-')

                c_descs.append(c_desc)
            # plt.show()
            # break

            # print c_desc
            c_descriptors.append(c_descs)

        log('Frames filled from neighbours: ' + str(frames_filled) + '/' +
            str(frames_total))
        self._descriptors = c_descriptors

    def get_clut_list(self, V):
        """
        Creates clustering distirbution and index list of each clustering
        Params
        V: linkage/KMeans object
        """
        clut_list = []
        clut_indices = []
        for nc in range(0, self._n_clusters):
            clut_indices.append(np.where(V == nc)[0])
        # print 'Clustering distirbution'
        # print '-----------------------'
        clut_list.append(clut_indices)
        for pos, c in enumerate(clut_list):
            obv_dev = []
            for nc in range(0, self._n_clusters):
                obv_dev.append((nc, len(c[nc])))
            sort_obd_dev = sorted(obv_dev, key=lambda x: x[1], reverse=True)
            # print sort_obd_dev
        self._clustering_dist = sort_obd_dev
        self._index_list = clut_list

    # def plot_cluster_distirbution(self, outp=None):
    #     lens = []
    #     oc = oct2py.Oct2Py()
    #     for i in self._clustering_dist:
    #         lens.append(i[1])
    #     oc.push('lens', lens)
    #     oc.push('xlens', range(0, self._n_clusters))
    #     if outp is None:
    #         oc.eval('plot(xlens,lens)',plot_width='2048', plot_height='1536')
    #     else:
    #         oc.eval('plot(xlens,lens)',
    #                 plot_dir=outp, plot_name='clustering_frequency', plot_format='jpeg',
    #                 plot_width='2048', plot_height='1536')

    def centroids_distance(self, dataset, features_first=False):
        # Returns the distances of each cluster centroid from given
        # dataset in ascending order.
        items = dataset.get_items()
        if features_first:
            items = np.transpose(items)
        dists = [(x, np.linalg.norm(self._centroids[x]-items))
                 for x in range(0, self._n_clusters)]
        dists = sorted(dists, key=lambda x: x[1], reverse=False)
        return dists

    def desc_date(self,nc_subset):
        # Creates the desc_date attribute. desc_date is used for indexing purposes.
        # Due to the fact that the exported netCDF frames are artificially made, we need
        # to give them a "real" id in the form of date. The id of an artificially made frame
        # is the first date inside each frame.
        desc_date = []
        for pos,i in enumerate(self._descriptors):
            gvalue = nc_subset._dataset.variables[nc_subset._time_name][i[0][0]]
            sim_date = ""
            for gv in gvalue:
                sim_date += gv
            desc_date.append(sim_date)
        self._desc_date = desc_date
        return desc_date

    def CH_evaluation(self):
        return metrics.calinski_harabaz_score(self.get_items(), self._labels)

    def ari(self, labels_true):
        return metrics.adjusted_rand_score(labels_true, self._labels)

    def nmi(self, labels_true):
        return metrics.adjusted_mutual_info_score(labels_true, self._labels)

    def save(self, filename='Clustering_object.zip'):
        utils.save(filename, self)

    def load(self, filename='Clustering_object.zip'):
        self = utils.load(filename)

# Example use of Clustering class

# from Dataset_transformations import Dataset_transformations
# from netcdf_subset import netCDF_subset
# import numpy as np
# if __name__ == '__main__':
#     data_dict = netCDF_subset(
#         'test_modified.nc', [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
#     items = [data_dict.extract_data()]
#     items = np.array(items)
#     #print items.shape
#     ds = Dataset_transformations(items, 1000)
#     ds.twod_transformation()
#     ds.normalize()
#     times = data_dict.get_times()
#     clust_obj = Clustering(ds, n_clusters=14, n_init=1, features_first=True)
#     clust_obj.kmeans()
#     # print clust_obj._labels.shape
#     clust_obj.create_density_descriptors(12, times)
#     # clust_obj.create_descriptors(12)
#     # print np.array(np.array(clust_obj._descriptors)[0][0]).shape
#     print clust_obj._descriptors.shape
