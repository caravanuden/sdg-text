import os

DATA_DIR = "data"
INPUT_DATA_DIR = os.path.join(DATA_DIR, "wikipedia")
INTERMEDIATE_SENTENCE_DATA_DIR = os.path.join(
    DATA_DIR, "intermediate_sentence_embeddings"
)
OUTPUT_SENTENCE_DATA_DIR = os.path.join(DATA_DIR, "sentence_embeddings")
OUTPUT_DOCUMENT_DATA_DIR = os.path.join(DATA_DIR, "document_embeddings")

LABEL_METADATA_PATH = os.path.join(DATA_DIR, "dhs_final_labels.csv")

COUNTRIES = [
    "AL",
    "BD",
    "CD",
    "CM",
    "GH",
    "GU",
    "HN",
    "IA",
    "ID",
    "JO",
    "KE",
    "KM",
    "LB",
    "LS",
    "MA",
    "MB",
    "MD",
    "MM",
    "MW",
    "MZ",
    "NG",
    "NI",
    "PE",
    "PH",
    "SN",
    "TG",
    "TJ",
    "UG",
    "ZM",
    "ZW",
    "BF",
    "BJ",
    "BO",
    "CO",
    "DR",
    "GA",
    "GN",
    "GY",
    "HT",
    "NM",
    "SL",
    "TD",
    "TZ",
    "AM",
    "AO",
    "BU",
    "CI",
    "EG",
    "ET",
    "KH",
    "KY",
    "ML",
    "NP",
    "PK",
    "RW",
    "SZ",
]
TARGETS = ["asset_index", "sanitation_index", "water_index", "women_edu"]
