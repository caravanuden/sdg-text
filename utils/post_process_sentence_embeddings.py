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
    metadata = pd.read_csv(LABEL_METADATA_PATH)
    metadata = ids.merge(metadata, on=key, how="left")[[key, target]]

    locations = metadata.DHSID_EA.unique()
    intermediate_embeddings = np.load(os.path.join(input_dir, "embeddings.npy"))

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

    consolidated_metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    np.save(
        os.path.join(output_dir, "embeddings.npy"), consolidated_embeddings,
    )


def post_process_sentence_embeddings_all(input_dir, output_dir, key="DHSID_EA"):
    ids = [
        pd.read_csv(os.path.join(input_dir, target, "metadata.csv"))
        for target in TARGETS
    ]
    ids = pd.concat(ids)[key].to_frame()

    metadata = pd.read_csv(LABEL_METADATA_PATH)
    metadata = ids.merge(metadata, on=key, how="left")[[key] + TARGETS]

    locations = metadata.DHSID_EA.unique()
    consolidated_metadata = metadata.drop_duplicates(subset=[key])

    intermediate_embeddings = [
        np.load(os.path.join(input_dir, target, "embeddings.npy")) for target in TARGETS
    ]
    intermediate_embeddings = np.concatenate(intermediate_embeddings, axis=0)
    consolidated_embeddings = np.zeros(
        (len(locations), intermediate_embeddings.shape[1])
    )
    for i, location in enumerate(locations):
        j = metadata[key] == location
        consolidated_embeddings[i, :] = np.mean(
            intermediate_embeddings[metadata[key] == location], axis=0
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

        if os.path.exists(os.path.join(OUTPUT_SENTENCE_DATA_DIR, country)):
            print(country, "all")
            post_process_sentence_embeddings_all(
                input_dir=os.path.join(OUTPUT_SENTENCE_DATA_DIR, country),
                output_dir=os.path.join(OUTPUT_SENTENCE_DATA_DIR, country, "all"),
            )
