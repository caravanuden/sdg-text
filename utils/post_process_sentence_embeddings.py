import numpy as np
import pandas as pd
import os
from constants import (
    INTERMEDIATE_SENTENCE_DATA_DIR,
    OUTPUT_SENTENCE_DATA_DIR,
    LABEL_METADATA_PATH,
    COUNTRIES,
    TARGETS,
    SENTENCE_EMBEDDING_SIZE,
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


def get_concatenated_sentence_embeddings(input_dir, output_dir, key="DHSID_EA"):
    intermediate_metadata = []
    intermediate_embeddings = []
    for target in TARGETS:
        if os.path.exists(os.path.join(input_dir, target)):
            target_metadata = pd.read_csv(
                os.path.join(input_dir, target, "metadata.csv")
            )
            target_embeddings = np.load(
                os.path.join(input_dir, target, "embeddings.npy")
            )
        else:
            target_metadata = pd.DataFrame(columns=[key])
            target_embeddings = []

        intermediate_metadata.append(target_metadata)
        intermediate_embeddings.append(target_embeddings)

    metadata = pd.concat(intermediate_metadata, axis=0).drop_duplicates(subset=[key])
    locations = metadata[key].unique()
    idxs = np.empty((len(locations), len(TARGETS)))
    for i, location in enumerate(locations):
        for j in range(len(TARGETS)):
            target_metadata = intermediate_metadata[j]
            if target_metadata.shape[0] > 0:
                matching_idxs = target_metadata[
                    target_metadata[key] == location
                ].index.values
                idxs[i, j] = matching_idxs[0] if matching_idxs else np.nan

    embeddings = np.zeros((len(locations), SENTENCE_EMBEDDING_SIZE * len(TARGETS),))
    for i, location in enumerate(locations):
        embeddings_for_loc = []
        for j in range(len(TARGETS)):
            if np.isnan(idxs[i, j]):
                embedding_for_target_and_loc = np.zeros((SENTENCE_EMBEDDING_SIZE))
            else:
                target_embeddings = intermediate_embeddings[j]
                embedding_for_target_and_loc = target_embeddings[int(idxs[i, j]), :]
            embedding_for_target_and_loc = np.expand_dims(
                embedding_for_target_and_loc, 0
            )
            embeddings_for_loc.append(embedding_for_target_and_loc)
        embeddings[i, :] = np.concatenate(embeddings_for_loc, axis=1)

    print(embeddings.shape)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    metadata.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)

    np.save(
        os.path.join(output_dir, "embeddings.npy"), embeddings,
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

        get_concatenated_sentence_embeddings(
            input_dir=os.path.join(OUTPUT_SENTENCE_DATA_DIR, country),
            output_dir=os.path.join(OUTPUT_SENTENCE_DATA_DIR, country, "concatenated"),
        )
