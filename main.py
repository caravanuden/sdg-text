"""
Next up is to programmatically import the files so we can run the experiments without heinous amounts of manual
intervention.
To do that, use the factory function at this link: https://stackoverflow.com/questions/41678073/import-class-from-module-dynamically
Basically, we jsut need a new modelInterfact that will take this _model as a module and then we can run the usual
training and testing code.

Almost there!

"""


from experiments.experiment import *
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import r2_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from models.feedforward_network import FeedforwardNewtork
from models.feedforward_network_with_nas import FeedforwardNetworkModuleForNAS, FeedforwardNetworkForNASModelInterface
from NAS.run_nas import run_nas

import argparse

import itertools

classification_cutoff_dict = {'asset_index': 0, 'sanitation_index': 3, 'water_index': 3, 'women_edu': 5}

TARGETS = ['asset_index', 'sanitation_index', 'water_index', 'women_edu']

feature_types = [FeatureType.target_sentence, FeatureType.all_sentence, FeatureType.document,
                 FeatureType.target_all_sentence, FeatureType.target_sentence_document,
                 FeatureType.all_sentence_document]

features = ['target_sentence', 'all_sentence', 'document']

model_types = [ModelType.regression, ModelType.classification]

metrics = {
    ModelType.regression: {r2_score},
    ModelType.classification: {classification_report, confusion_matrix, roc_auc_score, accuracy_score}
}

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

def run_experiments():

    # models = [Ridge(), LogisticRegression()]
    """models = [FeedforwardNewtork(hidden_dims=[100], output_dim=1,
                                 num_epochs=50, learning_rate=0.15),
              FeedforwardNewtork(hidden_dims=[100], output_dim=1,
                                 num_epochs=50, learning_rate=0.15, model_type=ModelType.classification)
              ]"""
    models = [FeedforwardNewtork(hidden_dims=[300, 200, 100], output_dim=1,
                                 num_epochs=50, learning_rate=0.15, model_type=ModelType.classification)]
    # model_types = [ModelType.regression, ModelType.classification]
    model_types = [ModelType.classification]

    # model_types = [ModelType.regression, ModelType.classification]


    experiment = Experiment(classification_cutoff_dict=classification_cutoff_dict,
                            targets=TARGETS, features=features, model_types=model_types,
                            models=models, metrics=metrics)
    experiment.run_experiments()


def run_nas_experiments():
    """
    This function trains and evals the models whose architecture was selected by NAS to see how they perform.
    :return:
    """
    exp = Experiment(classification_cutoff_dict=classification_cutoff_dict,
                            targets=TARGETS, features=features, metrics=metrics,
                     nas_model_interface=FeedforwardNetworkForNASModelInterface)

    exp.run_experiments_with_nas_selected_models()


def get_nas_selected_models():
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_combo', nargs='?', type=str)
    parser.add_argument('--target', nargs='?', type=str)
    parser.add_argument('--model_type', nargs='?', type=str)
    args = parser.parse_args()

    model_type = None
    if args.model_type == "classification":
        model_type = ModelType.classification
    if args.model_type == "regression":
        model_type = ModelType.regression

    feature_combo=None
    if args.feature_combo == "target_sentence":
        feature_combo = ["target_sentence"]
    elif args.feature_combo == "all_sentence":
        feature_combo = ["all_sentence"]
    elif args.feature_combo == "document":
        feature_combo = ["document"]
    elif args.feature_combo == "target_all_sentence":
        feature_combo = ["target_sentence", "all_sentence"]
    elif args.feature_combo == "target_sentence_document":
        feature_combo = ["target_sentence", "document"]
    elif args.feature_combo == "all_sentence_document":
        feature_combo = ["all_sentence", "document"]
    elif args.feature_combo == "all":
        feature_combo = features

    run_nas(model_class=FeedforwardNetworkModuleForNAS, target=args.target, features=feature_combo, model_type=model_type,
            classification_threshold=classification_cutoff_dict[args.target])

    """
    for model_type in model_types:
        for target in TARGETS:
            for feature_combo in powerset(features):
                feature_combo = list(feature_combo)
                if len(feature_combo) > 0:
                    run_nas(model_class=FeedforwardNetworkModuleForNAS, target=target, features=feature_combo, model_type=model_type,
                            classification_threshold = classification_cutoff_dict[target])
    """

if __name__ == "__main__":
    #nas_experiment()
    #run_nas_experiments()
    run_experiments()
