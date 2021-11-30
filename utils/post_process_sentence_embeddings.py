import numpy as np
import pandas as pd
import os
from utils.constants import (
    INTERMEDIATE_DATA_DIR,
    OUTPUT_DATA_DIR,
    LABEL_METADATA_PATH,
    COUNTRIES,
    TARGETS,
)


def postprocess_sentence_embeddings(input_dir, output_dir, key="DHSID_EA"):
    ids = pd.read_csv(os.path.join(input_dir, "metadata.csv"))
    metadata = pd.read_csv(LABEL_METADATA_PATH)
    metadata = ids.merge(metadata, on=key, how="left")[[key, target]]

    locations = metadata.DHSID_EA.unique()
    intermediate_embeddings = np.load(
        os.path.join(input_data_dir, country, target, "embeddings.npy")
    )

    consolidated_metadata = metadata.drop_duplicates(subset=[key])
    consolidated_embeddings = np.zeros(
        (len(locations), intermediate_embeddings.shape[1])
    )
    for i, location in enumerate(locations):
        consolidated_embeddings[i, :] = np.mean(
            intermediate_embeddings[metadata[key] == location], axis=0
        )

    print(intermediate_embeddings.shape, consolidated_embeddings.shape)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    consolidated_metadata.to_csv(os.path.join(output_dir, "metadata.csv"))

    np.save(
        os.path.join(output_dir, "embeddings.npy"), consolidated_embeddings,
    )


if __name__ == "__main__":
    for country in COUNTRIES:
        for target in TARGETS:
            if os.path.exists(os.path.join(INTERMEDIATE_DATA_DIR, country, target)):
                print(country, target)
                postprocess_sentence_embeddings(
                    input_dir=os.path.join(INTERMEDIATE_DATA_DIR, country, target),
                    output_dir=os.path.join(OUTPUT_DATA_DIR, country, target),
                )
