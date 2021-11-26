# Multimodal-deep-learning-for-poverty-prediction
#### TODOs
See (here)[https://docs.google.com/spreadsheets/d/17CHcM3IwEVwuU1_t8eH-G13271RGbywtcusSSUAz6ZA/edit?usp=sharing].

#### Data
From the SustainBench dataset, we have 117644 survey results from the years 1996-2019. These survey results cover 6 targets:
- sanitation index
- water index
- child mortality
- women education
- women BMI
- asset index (wealth/poverty)

For each data source of:
- Wikipedia
- GDELT
we have the following embeddings:
- Doc2Vec document embedding
- Sentence embedding for each target, gathered by extracting target-relevant sentences (sentences with words related to each target)
- Overall average sentence embedding (average or concat the separate sentence embeddings for the 6 targets)

#### Models
For the following models:
- Ridge regression
- Random forest?
- Feedforward DL regression
Train for the following (feature, target pairs):
- (Wikipedia Doc2Vec document embedding, each of the targets)
- (GDELT Doc2Vec document embedding, each of the targets)
- (Wiki + GDELT Doc2Vec document embedding, each of the targets)
- (Wikipedia overall sentence embedding, each of the targets)
- (GDELT overall sentence embedding, each of the targets)
- (Wiki + GDELT overall sentence embedding, each of the targets)
- For each target:
  - (Wikipedia {target} sentence embedding, {target})
  - (GDELT {target} sentence embedding, {target})
  - (Wiki + GDELT {target} sentence embedding, {target})
