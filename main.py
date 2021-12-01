from experiments.experiment import *
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, classification_report, confusion_matrix
from models.feedforward_network import FeedforwardNewtork


if __name__ == "__main__":
    classification_cutoff_dict = {'asset_index': 0, 'sanitation_index': 3, 'water_index': 3, 'women_edu': 5}

    TARGETS = ['asset_index', 'sanitation_index', 'water_index', 'women_edu']

    feature_types = [FeatureType.target_sentence, FeatureType.all_sentence, FeatureType.document,
                     FeatureType.target_all_sentence, FeatureType.target_sentence_document,
                     FeatureType.all_sentence_document]

    #models = [Ridge(), LogisticRegression()]
    models = [FeedforwardNewtork(hidden_dims=[100], output_dim=1)]
    model_types = [ModelType.regression]

    #model_types = [ModelType.regression, ModelType.classification]


    metrics = {
        ModelType.regression: {r2_score},
        ModelType.classification: {classification_report, confusion_matrix}
    }


    experiment = Experiment(classification_cutoff_dict=classification_cutoff_dict,
                            targets=TARGETS, feature_types=feature_types, model_types=model_types,
                            models=models, metrics=metrics)
    experiment.run_experiments()