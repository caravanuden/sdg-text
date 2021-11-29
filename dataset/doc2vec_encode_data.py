# the purpose of this file is to:
# 1) get only those wiki articles whose lat/lon coordinates have a corresponding image in the database
# 2) doc2vec encode those articles

from Utility import *
import os
import zipfile
import json
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from preprocess_wiki_data import *
import argparse


LAT_LONG_ACCETABLE_MATCHING_RADIUS = 6



class GensimDocumentsIterator():
    def __init__(self, verbose, save_memory):
        """
        :param verbose (bool): whether or not to print out error messages when loading wiki files
        (usually, these are errors in parsing the lat/lon values from the geolcated aritcles' scraped coordinates)
        :param save_memory (bool): if this is true, then we only use geolocated articles that are relevant to DHS when
        training the Doc2Vec model. If False, we use ALL geolocated articles (even those which don't match a location in
        DHS).
        """
        self.verbose = verbose
        self.save_memory = save_memory
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


    def initialize_preprocessed_data(self, use_wiki=True, use_gdelt=True):
        """

        :param verbose:
        :return: This function will preprocess the data and write to files if it doesn't already exist
        """
        if use_wiki:
            self.wiki_initializer.init_wiki_data(self.verbose)
        if use_gdelt:
            pass # temporary



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

        # only try and build the actual files from the raw, scraped data the first time through the iterator
        if not self.wiki_initializer.all_files_ready():
            self.initialize_preprocessed_data()

        # now, just read in the data
        zip_file_names = [file_name for file_name in os.listdir(PATH_TO_PREPROCESSED_DOC2VEC_INPUTS) if
                          os.path.splitext(file_name)[1] == ".zip"]
        for i,zip_file_name in enumerate(zip_file_names):
            print("***training on ZIP FILE file {} out of {} total***".format(i, len(zip_file_names) - 1))
            zip_file = zipfile.ZipFile(os.path.join(PATH_TO_PREPROCESSED_DOC2VEC_INPUTS, zip_file_name))
            file_names = zip_file.namelist()
            for file_name in file_names:
                file = zip_file.read(file_name)
                file_json = json.loads(file.decode())  # call decode because file.read() is just bytes
                for document in file_json:
                    tag = document["tag"]
                    if self.save_memory and "None" in tag:
                        continue
                    else:
                        # only yield document if it has a valid DHS ID.
                        yield TaggedDocument(words=document["clean_text"].split(), tags=[document["tag"]])


def get_doc2vec_output_model_name(args):
    prefix = "wiki_trained_doc2vec_model"
    file_name = "_".join([prefix, str(args.epochs), "epochs", str(args.save_memory), "save_memory"]) + ".model"
    return os.path.join(PATH_TO_DOC2VEC_MODEL, file_name)



def doc2vec_encode():
    """
    :return: In this function, we'll train the doc2vec model on the relevant articles (Those with matching lat/long values in the dataset
    Then, we'll save the resulting doc2vec model and, essentially, use that for the dataloader stuff
    """
    # get number of epochs to train with
    # get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', nargs='?', type=int, default=1)
    parser.add_argument('--save_memory', action='store_false') # deafult is to have save_memory be true
    parser.add_argument('--verbose', action='store_true')  # deafult is to have verbose be false
    args = parser.parse_args()


    data_iter = GensimDocumentsIterator(args.verbose, args.save_memory)
    model = Doc2Vec(documents=data_iter, vector_size=300, window=8, min_count=1, workers=4, epochs=args.epochs)

    print("training doc2vec model...")
    model.train(data_iter, total_examples=model.corpus_count)
    print("finished training doc2vec model.")

    # now, we'll save it
    model.save(get_doc2vec_output_model_name(args))


if __name__ == "__main__":
    doc2vec_encode()



