import urllib.request
import re
import os
import bz2
import mwxml
import multiprocessing

# what we should od:
# use the base url and then use that to find all lnks with downloads to the pages thing
# then, once we have all htose file names, download them sequentially and use mwxml to parse them.
# easy peasy!

BASE_URL = "https://dumps.wikimedia.org/enwiki/latest/"
ARTICLE_XML_REGEX = "\"(enwiki-latest-pages-articles[0-9]+.xml-.*.bz2)\""

def extract_location_data():
    pass


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
        for revision in page:
            new_page_location = extract_location_data(revision.text)
            if new_page_location is not None:
                page_location = new_page_location

        if page_location is not None:
            yield page.title, page_location


def get_dump():
    # first, get a list of the xml file names that have article information and text
    page_text = urllib.request.urlopen(BASE_URL).read()
    xml_file_names = re.findall(page_text, ARTICLE_XML_REGEX)

    NUM_PARALLEL_PROCESSES = 4

    # next, get NUM_PARALLEL_PROCESSES articles downloaded parallely
    # and then parse them in parallel
    q = multiprocessing.Queue()
    pools_of_xml_file_names = chunks(xml_file_names, NUM_PARALLEL_PROCESSES)

    for pool_of_xml_file_names in pools_of_xml_file_names:
        for xml_file_name in pool_of_xml_file_names:
            p = multiprocessing.Process(target=urllib.request.urlretrieve,
                                        args=(os.path.join(BASE_URL, xml_file_name), xml_file_name))
            p.start()
        print(q.get())  # prints "[42, None, 'hello']"
        p.join()


    # next, get the articles one at at a time and parse them
    for xml_file_name in xml_file_names:
        urllib.request.urlretrieve(os.path.join(BASE_URL, xml_file_name), xml_file_name)

        # then, parse the file
        dump = mwxml.Dump.from_file(bz2.open(xml_file_name))
        process_dump(dump)





test = mwxml.Dump.from_file(bz2.open("enwiki-latest-pages-articles1.xml-p1p41242.bz2"))
x = 5
for page in test:
    y=6
    #test2 = sum(1 for _ in page)
    for revision in page:
        z = 7
