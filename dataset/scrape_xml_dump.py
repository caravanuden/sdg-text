import urllib.request
import re
import os
import bz2
import mwxml

# what we should od:
# use the base url and then use that to find all lnks with downloads to the pages thing
# then, once we have all htose file names, download them sequentially and use mwxml to parse them.
# easy peasy!

BASE_URL = "https://dumps.wikimedia.org/enwiki/latest/"
ARTICLE_XML_REGEX = "\"(enwiki-latest-pages-articles[0-9]+.xml-.*.bz2)\""

def extract_location_data():
    pass


def process_dump(dump):
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
