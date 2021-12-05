import numpy as np
import pandas as pd
import os
from constants import (
    INTERMEDIATE_SENTENCE_DATA_DIR,
    OUTPUT_SENTENCE_DATA_DIR,
    LABEL_METADATA_PATH,
    COUNTRIES,
    TARGETS,
)


def post_process_sentence_embeddings(input_dir, output_dir, key="DHSID_EA"):
    ids = pd.read_csv(os.path.join(input_dir, "metadata.csv"))
    consolidated_metadata = ids.drop_duplicates(subset=[key])
    locations = consolidated_metadata[key].unique()

    intermediate_embeddings = np.load(os.path.join(input_dir, "embeddings.npy"))

    consolidated_embeddings = np.zeros(
        (len(locations), intermediate_embeddings.shape[1])
    )
    for i, location in enumerate(locations):
        consolidated_embeddings[i, :] = np.mean(
            intermediate_embeddings[ids[key] == location], axis=0
        )

    print(intermediate_embeddings.shape, consolidated_embeddings.shape)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    consolidated_metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    np.save(
        os.path.join(output_dir, "embeddings.npy"), consolidated_embeddings,
    )


if __name__ == "__main__":
    for country in COUNTRIES:
        for target in TARGETS:
            if os.path.exists(
                os.path.join(INTERMEDIATE_SENTENCE_DATA_DIR, country, target)
            ):
                print(country, target)
                post_process_sentence_embeddings(
                    input_dir=os.path.join(
                        INTERMEDIATE_SENTENCE_DATA_DIR, country, target
                    ),
                    output_dir=os.path.join(OUTPUT_SENTENCE_DATA_DIR, country, target),
                )
