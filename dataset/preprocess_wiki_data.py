# the purpose of this file is to:
# 1) get only those wiki articles whose lat/lon coordinates have a corresponding image in the database
# 2) doc2vec encode those articles

from Utility import *


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

    coordinate_data = coord_tag.split("|")[1:]

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






def doc2vec_encode():
    """
    :return: In this function, we'll train the doc2vec model on the relevant articles (Those with matching lat/long values in the dataset
    Then, we'll save the resulting doc2vec model and, essentially, use that for the dataloader stuff
    """
