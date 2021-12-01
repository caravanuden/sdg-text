from experiments.experiment import *
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, classification_report



if __name__ == "__main__":
    classification_cutoff_dict = {'asset_index': 0, 'sanitation_index': 3, 'water_index': 3, 'women_edu': 5}

    TARGETS = ['asset_index', 'sanitation_index', 'water_index', 'women_edu']

    feature_types = [FeatureType.target_sentence, FeatureType.all_sentence, FeatureType.document,
                     FeatureType.target_all_sentence, FeatureType.target_sentence_document,
                     FeatureType.all_sentence_document]

    models = [Ridge(), LogisticRegression()]

    model_types = [ModelType.regression, ModelType.classification]

    metrics = {
        ModelType.regression: {r2_score},
        ModelType.classification: {classification_report}
    }


    experiment = Experiment(classification_cutoff_dict, TARGETS, feature_types, model_types, models, metrics)
    experiment.run_experiments()