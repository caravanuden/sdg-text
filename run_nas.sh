#!/bin/bash

feature_combos=("target_sentence" "all_sentence" "document" "target_all_sentence" "target_sentence_document" "all_sentence_document" "all")
model_types=("classification" "regression")
targets=("asset_index" "sanitation_index" "water_index" "women_edu")

for i in "${model_types[@]}"
do
   for j in "${targets[@]}"
   do
        for k in "${feature_combos[@]}"
        do
            nohup python3 -u main.py --feature_combo=$k --target=$j --model_type=$i &
            wait
        done
   done
done