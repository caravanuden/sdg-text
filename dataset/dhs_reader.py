from Utility import *
import csv
import numpy as np


class DHSReader:
    def __init__(self, acceptable_radius=4):
        self.dhs_ids, self.dhs_lat_long_np_array = self.read_in_dhs_data()
        self.acceptable_radius = acceptable_radius
        self.id_counter = 0

    def read_in_dhs_data(self):
        """

        :return: (list, nparray) list of ids and their corresponding lat-long values in an np array.
          to be used to find the correct id corresponding to each wiki article's geolocation
        """
        dhs_labels_csv = open(PATH_TO_DHS_LABELS, 'r')
        dhs_labels_csv_reader = csv.reader(dhs_labels_csv, delimiter=',')

        lat_long_list = list()
        ids_list = list()

        for i, dhs_label_row in enumerate(dhs_labelnps_csv_reader):
            if i == 0:
                continue # skip the row giving the column titles
            curr_id = dhs_label_row[0]
            latitude = float(dhs_label_row[3])
            longitude = float(dhs_label_row[4])

            ids_list.append(curr_id)
            lat_long_list.append([latitude,longitude])

        return ids_list, np.array(lat_long_list)

    def get_closest_dhs_ids_to_lat_lon(self, latitude, longitude): #, top_k = None):
        """

        :param latitude:
        :param longitude: we'll use thes
        #:param top_k (int): the number of indices to return (return all k minimimum ones)
        if None, return a single value
        # :return: list of ints: the list of indices (sorted) that are within the acceptable radius to the given location
          :return: (string) the id of the dhs row that is closest to the provided lat/lon value.
        """
        lat_long_vector = np.array([latitude, longitude]).reshape(1, -1)
        norms = np.linalg.norm(self.dhs_lat_long_np_array - lat_long_vector, axis=1)
        min_index = np.argmin(norms)
        if norms[min_index] <= self.acceptable_radius:
            return self.dhs_ids[min_index]
        return None


