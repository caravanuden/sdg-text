from Utility import *
import os
import zipfile
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import csv
import numpy as np


def clean(text):
    """

    :param text (string): supposed to be text from a Wiki article
    :return: the text cleaned up.
    """

    # temporary
    return text



def get_lat_lon_from_wiki_coord_tag(coord_tag):
    """

    :param coord_tag (string): a coordinate tag of the type mentioned here: https://en.wikipedia.org/wiki/Template:Coord#To_extract_the_latitude_from_a_Coord_template
    In particular, these come in the following forms:

    Case 1: {{coord|dd|N/S|dd|E/W|coordinate parameters|template parameters}}
    Case 2: {{coord|dd|mm|N/S|dd|mm|E/W|coordinate parameters|template parameters}}
    Case 3: {{coord|dd|mm|ss|N/S|dd|mm|ss|E/W|coordinate parameters|template parameters}}
    Case 4: {{coord|latitude|longitude|coordinate parameters|template parameters}}
    :return: (tuple): (latitude,longitude) value parsed from the coordinate tag.

    Note that this function is basically a python equivalent of the function written in Lua documented here: https://en.wikipedia.org/wiki/Module:Coordinates
    Actually ,that's no longer true; I'm not sure what that function is doing, but it seems like they're somehow pre-processing the coord string beforehand or seomthing?
    This function is pretty gross; probably a better way to write it.
    """
    def convert_degrees_minutes_seconds_lat_or_long(degrees, minutes, seconds, direction):
        return (degrees + minutes / 60 + seconds / (60 * 60)) * (-1 if direction in ['W', 'S'] else 1)

    #coordinate_data = coord_tag.split("|")[1:]
    filter_outs = ["display=title", "display=inline,title", "format=dms"]
    coordinate_data = [item.strip() for item in coord_tag.split("|") if not item.isspace() and item not in filter_outs]

    lat_degrees = lat_minutes = lat_seconds = 0
    long_degrees = long_minutes = long_seconds = 0
    lat_direction = long_direction = None

    latitude = longitude = None

    if len(coordinate_data) < 4:
        # then, we have Case 4.
        latitude = float(coordinate_data[0])
        longitude = float(coordinate_data[1])
    elif coordinate_data[1] == "N" or coordinate_data[1] == "S":
        # then, we have Case 1
        lat_degrees = float(coordinate_data[0])
        lat_direction = coordinate_data[1]

        long_degrees = float(coordinate_data[2])
        long_direction = coordinate_data[3]
    elif coordinate_data[2] == "N" or coordinate_data[2] == "S":
        # then, we have Case 2
        lat_degrees = float(coordinate_data[0])
        lat_minutes = float(coordinate_data[1])
        lat_direction = coordinate_data[2]

        long_degrees = float(coordinate_data[3])
        long_minutes = float(coordinate_data[4])
        long_direction = coordinate_data[5]

    elif coordinate_data[3] == "N" or coordinate_data[3] == "S":
        # then, we have Case 3
        lat_degrees = float(coordinate_data[0])
        lat_minutes = float(coordinate_data[1])
        lat_seconds = float(coordinate_data[2])
        lat_direction = coordinate_data[3]

        long_degrees = float(coordinate_data[4])
        long_minutes = float(coordinate_data[5])
        long_seconds = float(coordinate_data[6])
        long_direction = coordinate_data[7]
    else:
        # there was a bug when we obtained the caputre groups from the regular expression
        # so we can also try this as a last resort
        #latitude = float(coordinate_data[0])
        #longitude = float(coordinate_data[1])

        #print("Set lat/lon using else branch with coordinate_tag={}".format(coord_tag))

        raise Exception("The coordinate tag is invalid.")


    #else:
    #    raise Exception("Coordinate is not parsable.")
    if latitude is None:
        latitude = convert_degrees_minutes_seconds_lat_or_long(lat_degrees, lat_minutes, lat_seconds, lat_direction)
    if longitude is None:
        longitude = convert_degrees_minutes_seconds_lat_or_long(long_degrees, long_minutes, long_seconds, long_direction)

    return (latitude,longitude)

class WikiInitializer:
    def __init__(self, acceptable_radius = 4):
        self.dhs_ids = None
        self.dhs_lat_long_np_array = None
        if not self.all_files_ready():
            self.dhs_ids, self.dhs_lat_long_np_array = self.read_in_dhs_data()
        self.acceptable_radius=4
        self.id_counter = 0

    def all_files_ready(self):
        """

        :return: (bool) indicates whether or not all the preprocessed wiki files are created already or not.
        """
        # get all the zip files in the Wikipedia outputs directory
        zip_file_names = [file_name for file_name in os.listdir(PATH_TO_WIKIPEDIA_OUTPUTS) if
                          os.path.splitext(file_name)[1] == ".zip"]
        for i, zip_file_name in enumerate(zip_file_names):
            zip_file = zipfile.ZipFile(os.path.join(PATH_TO_WIKIPEDIA_OUTPUTS, zip_file_name))
            file_names = zip_file.namelist()
            for j, file_name in enumerate(file_names):
                if not os.path.isfile(output_file_path):
                    return False

        return True


    def read_in_dhs_data(self):
        """

        :return: (list, nparray) list of ids and their corresponding lat-long values in an np array.
          to be used to find the correct id corresponding to each wiki article's geolocation
        """
        dhs_labels_csv = open(PATH_TO_DHS_LABELS, 'r')
        dhs_labels_csv_reader = csv.reader(dhs_labels_csv, delimiter=',')

        lat_long_list = list()
        ids_list = list()

        for i, dhs_label_row in enumerate(dhs_labels_csv_reader):
            id = dhs_label_row[0]
            lat = float(dhs_label_row[3])
            long = float(dhs_label_row[4])

            ids_list.append(id)
            lat_long_list.append([lat,long])

        return ids_list, np.array(lat_long_list)

    def get_tag(self, latitude, longitude):
        """
        :param latitude:
        :param longitude:
        :return: search through the dhs labels and find label with closest latitude,longitude values.
         Then, make the tag for the document embedding that row's ID.
        """
        lat_long_vector = np.array([latitude, longitude]).reshape(1,-1)
        norms = np.linalg.norm(self.dhs_lat_long_np_array - lat_long_vector)
        min_index = np.argmin(norms)
        if norms[min_index] <= self.acceptable_radius:
            return f"{self.dhs_ids[min_index]}-wiki" # we add wiki to the end to diffferentiate the wiki and gdelt data

        self.id_counter += 1
        return f"no_id_{self.id_counter}-wiki" # these doc vectors won't be matched with an article


    def init_wiki_data(self, verbose=False):
        # get all the zip files in the Wikipedia outputs directory
        zip_file_names = [file_name for file_name in os.listdir(PATH_TO_WIKIPEDIA_OUTPUTS) if
                          os.path.splitext(file_name)[1] == ".zip"]
        num_exceptions = 0
        for i, zip_file_name in enumerate(zip_file_names):
            # read the zip files without actually extracting them
            print("***working on ZIP file {} out of {} total***".format(i, len(zip_file_names) - 1))
            print("{} total exceptions so far".format(num_exceptions))
            zip_file = zipfile.ZipFile(os.path.join(PATH_TO_WIKIPEDIA_OUTPUTS, zip_file_name))
            file_names = zip_file.namelist()

            output_json = list()

            # create the file if we need to only.
            # if not os.path.is_file(output_file_path):
            for j, file_name in enumerate(file_names):
                # print("working on file {} out of {} total".format(j, len(file_names) - 1))
                output_file_path = os.path.join(PATH_TO_PREPROCESSED_DOC2VEC_INPUTS,
                                                "preprocessed_{}_{}.json".format(zip_file_name.split(".")[0], j))
                if not os.path.isfile(output_file_path):
                    file = zip_file.read(file_name)
                    file_json = json.loads(file.decode())  # call decode because file.read() is just bytes

                    for document in file_json:
                        coord_tag = document["page_location"].split(",")[0][
                                    3:-1]  # ugly, but necessary because of how the page_location tuple is json encoded. Oops!
                        try:
                            latitude, longitude = get_lat_lon_from_wiki_coord_tag(coord_tag)

                            # also clean the wiki page text
                            clean_page_text = clean(document["page_text"])


                            output_json.append({
                                "clean_text": clean_page_text,
                                "tag": self.get_tag(latitude, longitude)
                            })
                        except Exception as e:
                            num_exceptions += 1

                            if verbose:
                                print(
                                    "----\ngot exception: {} \n for coordinate tag: {} \n for page_loc: {} \n\n\n----".format(
                                        e, coord_tag, document["page_location"]))

                    # file.close()

                    writeToJsonFile(output_json, output_file_path)

            zip_file.close()