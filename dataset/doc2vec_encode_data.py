# the purpose of this file is to:
# 1) get only those wiki articles whose lat/lon coordinates have a corresponding image in the database
# 2) doc2vec encode those articles

from Utility import *
import os
import zipfile
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess_wiki_data import *


LAT_LONG_ACCETABLE_MATCHING_RADIUS = 4



class GensimDocumentsIterator():
    def __init__(self):
        self.wiki_initializer = WikiInitializer(LAT_LONG_ACCETABLE_MATCHING_RADIUS)

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


    def init_gdelt_data(self):
        pass
        #zip_file = zipfile.ZipFile(os.path.join(PATH_TO_GDELT_OUTPUTS, "gdelt_articles.zip"))
        #file =

        """
        file_names = zip_file.namelist()
        for j, file_name in enumerate(file_names):
            file = zip_file.read(file_name)
        """


    def initialize_preprocessed_data(self, verbose=False, use_wiki=True, use_gdelt=True):
        """

        :param verbose:
        :return: This function will preprocess the data and write to files if it doesn't already exist
        """
        if use_wiki:
            self.wiki_initializer.init_wiki_data(verbose)
        if use_gdelt:
            pass # temporary



    def load(self, verbose=False):
        """

        :return: This function yields a SINGLE article each time
        This function will basically read in all Wiki articles that are geolocated (the ones we
        obtained by scraping) and then use these articles to train a Doc2Vec model.
        Then, we can use the Doc2Vec embeddings as features in our late fusion model.
        For now, I think I'll try and train on ALL the Wiki articles, and we can decide later how we want
        to actually associate those articles with data points.

        For now, only load files from Wikipedia (can add in GDELT later)
        """

        # only try and build the actual files from the raw, scraped data the first time through the iterator
        if not self.wiki_initializer.all_files_ready():
            self.initialize_preprocessed_data(verbose)

        # now, just read in the data
        file_names = [file_name for file_name in os.listdir(PATH_TO_PREPROCESSED_DOC2VEC_INPUTS)]
        for i, file_name in enumerate(file_names):
            # read the zip files without actually extracting them
            if i == 0 or i % 10 == 0 or i == len(file_names)-1:
                print("***working on INPUT FILE file {} out of {} total***".format(i, len(file_names) - 1))
            file_json = readFromJsonFile(os.path.join(PATH_TO_PREPROCESSED_DOC2VEC_INPUTS, file_name))
            for document in file_json:
                yield TaggedDocument(words=document["clean_text"].split(), tags=[document["tag"]])


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
    model.save(os.path.join(PATH_TO_DOC2VEC_MODEL, "wiki_trained_doc2vec_model.model"))


if __name__ == "__main__":
    doc2vec_encode()



