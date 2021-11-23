# the purpose of this file is to:
# 1) get only those wiki articles whose lat/lon coordinates have a corresponding image in the database
# 2) doc2vec encode those articles

from Utility import *
import os
import zipfile
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

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
    coordinate_data = coord_tag.split("|")

    lat_degrees = lat_minutes = lat_seconds = 0
    long_degrees = long_minutes = long_seconds = 0
    lat_direction = long_direction = None

    latitude = longitude = None

    if len(coordinate_data) > 1 and coordinate_data[1] == "N" or coordinate_data[1] == "S":
        # then, we have Case 1
        lat_degrees = float(coordinate_data[0])
        lat_direction = coordinate_data[1]

        long_degrees = float(coordinate_data[2])
        long_direction = coordinate_data[3]
    elif len(coordinate_data) > 2 and coordinate_data[2] == "N" or coordinate_data[2] == "S":
        # then, we have Case 2
        lat_degrees = float(coordinate_data[0])
        lat_minutes = float(coordinate_data[1])
        lat_direction = coordinate_data[2]

        long_degrees = float(coordinate_data[3])
        long_minutes = float(coordinate_data[4])
        long_direction = coordinate_data[5]

    elif len(coordinate_data) > 3 and coordinate_data[3] == "N" or coordinate_data[3] == "S":
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
        # then, we have Case 4.
        latitude = float(coordinate_data[0])
        longitude = float(coordinate_data[1])

    #else:
    #    raise Exception("Coordinate is not parsable.")
    if latitude is None:
        latitude = convert_degrees_minutes_seconds_lat_or_long(lat_degrees, lat_minutes, lat_seconds, lat_direction)
    if longitude is None:
        longitude = convert_degrees_minutes_seconds_lat_or_long(long_degrees, long_minutes, long_seconds, long_direction)

    return (latitude,longitude)





class GensimDocumentsIterator():
    def __init__(self):
        pass

    def __iter__(self):
        # reset the generator
        self.generator = self.load()
        return self

    def __next__(self):
        result = next(self.generator)
        if result is None:
            raise StopIteration
        else:
            return result

    def load(self):
        """

        :return: This function yields a SINGLE article each time
        This function will basically read in all Wiki articles that are geolocated (the ones we
        obtained by scraping) and then use these articles to train a Doc2Vec model.
        Then, we can use the Doc2Vec embeddings as features in our late fusion model.
        For now, I think I'll try and train on ALL the Wiki articles, and we can decide later how we want
        to actually associate those articles with data points.

        For now, only load files from Wikipedia (can add in GDELT later)
        """

        # get all the zip files in the Wikipedia outputs directory
        # zip_file_names = os.listdir(PATH_TO_WIKIPEDIA_OUTPUTS)
        zip_file_names = [file_name for file_name in os.listdir(PATH_TO_WIKIPEDIA_OUTPUTS) if os.path.splitext(file_name)[1] == ".zip"]
        for zip_file_name in zip_file_names:
            # read the zip files without actually extracting them
            zip_file = zipfile.ZipFile(os.path.join(PATH_TO_WIKIPEDIA_OUTPUTS, zip_file_name))
            for file_name in zip_file.namelist():
                file = zip_file.read(file_name)
                file = json.loads(file.read().decode()) # call decode because file.read() is just bytes

                for document in file:
                    coord_tag = document["page_location"].split(",")[0][3:-1] # ugly, but necessary because of how the page_location tuple is json encoded. Oops!
                    latitude,longitude = get_lat_lon_from_wiki_coord_tag(coord_tag)

                    # also clean the wiki page text
                    clean_page_text = clean(document["page_text"])

                    # NOTE: we use the value document_latitude-document_longitude as the document tag!
                    # this is VERY IMPORTANT! The tag is what is used to obtain the desired document vector later on
                    # (i.e., to obtain a document at latitude,longitude value -30,40, we would use model.dv["-30-40"]
                    yield TaggedDocument(words=clean_page_text.split(), tags=["{}-{}".format(latitude, longitude)])

                file.close()

            zip_file.close()




def doc2vec_encode():
    """
    :return: In this function, we'll train the doc2vec model on the relevant articles (Those with matching lat/long values in the dataset
    Then, we'll save the resulting doc2vec model and, essentially, use that for the dataloader stuff
    """
    data_iter = GensimDocumentsIterator()
    model = Doc2Vec(documents=data_iter, vector_size=300, window=8, min_count=1, workers=4, epochs=10)

    print("training doc2vec model...")
    model.train(data_iter, total_examples=model.corpus_count, epochs=2)
    print("finished training doc2vec model.")

    # now, we'll save it
    model.save("wiki_trained_doc2vec_model.model")




