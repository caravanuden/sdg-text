import numpy as np
import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec
from constants import (
    INTERMEDIATE_SENTENCE_DATA_DIR,
    OUTPUT_SENTENCE_DATA_DIR,
    LABEL_METADATA_PATH,
    COUNTRIES,
    TARGETS,
    OUTPUT_DOCUMENT_DATA_DIR,
)


def post_process_document_embeddings(model_path, output_dir, key="DHSID_EA"):
    model = Doc2Vec.load(model_path)
    document_vectors = model.dv

    countries_ids = {country: [] for country in COUNTRIES}
    countries_vectors = {country: [] for country in COUNTRIES}
    for document_key in document_vectors.index_to_key:
        country = document_key[:2]
        if country not in COUNTRIES:
            pass
        id = "-".join(document_key.split("-")[:4])
        countries_ids[country].append(id)
        countries_vectors[country].append(document_vectors[document_key])

    metadata = pd.read_csv(LABEL_METADATA_PATH)
    for country in COUNTRIES:
        country_ids = pd.DataFrame(countries_ids[country], columns=["DHSID_EA"])
        country_vectors = np.stack(countries_vectors[country])
        print(country)
        print(country_vectors.shape)

        output_dir_for_country = os.path.join(output_dir, country)

        if not os.path.isdir(output_dir_for_country):
            os.makedirs(output_dir_for_country)

        country_ids.to_csv(
            os.path.join(output_dir_for_country, "metadata.csv"), index=False
        )

        np.save(
            os.path.join(output_dir_for_country, "embeddings.npy"), country_vectors,
        )


if __name__ == "__main__":
    post_process_document_embeddings(
        model_path="models/wiki_trained_doc2vec_model_10_epochs.model",
        output_dir=OUTPUT_DOCUMENT_DATA_DIR,
        key="DHSID_EA",
    )
