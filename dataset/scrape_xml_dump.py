import urllib.request
import re
import os
import mwxml
from multiprocessing.pool import ThreadPool

# what we should od:
# use the base url and then use that to find all lnks with downloads to the pages thing
# then, once we have all htose file names, download them sequentially and use mwxml to parse them.
# easy peasy!

# helpful link: https://github.com/mediawiki-utilities/python-mwxml/blob/master/ipython/labs_example.ipynb

BASE_URL = "https://dumps.wikimedia.org/enwiki/latest/"
ARTICLE_XML_REGEX = "\"(enwiki-latest-pages-articles-multistream[0-9]+.xml-.*.bz2)\""

def extract_location_data(text):
    """
    This function will be used to detect if a given revision contains location data for the page overall
    NOTE: We need to make sure this is the only way that location data is specified!
    :param text: the text from which we want to capture coordinate data
    :return: The capture groups from the text
    For example,
    extract_location_data(coordinates = {{Coord|12|31|07|N|70|02|09|W|type:city}})
    should return [(12|31|07|N|70|02|09|W, city)]
    """
    #regex = "{{coord\|([0-9]+\.[0-9]+\|[0-9]+\.[0-9]+)}}"
    regex = "coordinates\s*=\s*{{Coord\|(.*)\|type:(.*)}}"
    capture_groups = re.findall(regex, text)
    if len(capture_groups) == 0:
        return None
    return capture_groups



def chunks(array, n):
    """Yield successive n-sized chunks from array."""
    for i in range(0, len(array), n):
        yield array[i:i + n]


def process_dump(dump, path):
    """
    Note: this function will be used by wmxml to to read the xml wiki dump files in parallel and
    execute the function.
    :param dump: the mwxml dump object to be passed in
    :param path: the path of the file to read the xml from
    :return:
    """
    for page in dump:
        page_location = None
        article_text = None
        for revision in page:
            try:
                # there will be only one revision in the case of multi-stream; the latest revision and the text is the actual article text.
                new_page_location = extract_location_data(revision.text)
                if new_page_location is not None:
                    page_location = new_page_location
                    article_text = revision.text
            except Exception as e:
                print(e)


        if page_location is not None:
            yield page.title, page_location, article_text


def xml_parse_wikidump():
    """
    This function downloads and parses the xml files from the wikimedia dump.
    :return:
    """
    outputs = list()

    # first, get a list of the xml file names that have article information and text
    page_text = urllib.request.urlopen(BASE_URL).read().decode()
    xml_file_names = re.findall(ARTICLE_XML_REGEX, page_text)

    NUM_PARALLEL_PROCESSES = 4

    # next, get NUM_PARALLEL_PROCESSES articles downloaded parallely
    # and then parse them in parallel
    pools_of_xml_file_names = chunks(xml_file_names, NUM_PARALLEL_PROCESSES)
    for pool_of_xml_file_names in pools_of_xml_file_names:
        with ThreadPool(processes=NUM_PARALLEL_PROCESSES) as pool:
            pool.starmap(urllib.request.urlretrieve,
                     zip([os.path.join(BASE_URL, xml_file_name) for xml_file_name in pool_of_xml_file_names],
                         pool_of_xml_file_names))

        for page_name, page_location, page_text in mwxml.map(process_dump, pool_of_xml_file_names):
            outputs.append += "{}, {}, {}".format(page_name, page_location, page_text)

        for xml_file_name in pool_of_xml_file_names:
            os.remove(xml_file_name)

    output_file = open("outputs.txt", "w")
    output_file.write('\n'.join(outputs))
    output_file.close()


if __name__ == '__main__':
    xml_parse_wikidump()
