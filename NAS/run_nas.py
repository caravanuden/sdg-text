from utils.get_torch_data_loader import *
from utils.get_data_loader import *
import nni.retiarii.evaluator.pytorch.lightning as pl
import nni.retiarii.nn.pytorch as nn
from nni.retiarii import serialize
import nni.retiarii.strategy as strategy
from nni.retiarii.experiment.pytorch import RetiariiExeConfig, RetiariiExperiment, debug_mutated_model
from utils.constants import DATA_DIR
import random
import pdb
from experiments.experiment import ModelType
from utils.file_utils import *


def run_nas(model_class, target,features, num_hidden_layers_in_network_doing_nas=4, model_batch_size=32, model_epochs=10,
            model_type=ModelType.classification, classification_threshold=0):
    """

    :param model: should be the class inheriting from nni.retiarii.nn.pytorch.Module which has some
    Mutation Primitives (see here: https://nni.readthedocs.io/en/stable/NAS/MutationPrimitives.html)
    :return:
    """
    features_string = ",".join(features)
    outfile_name = f"{model_type.name}_{features_string}_{target}_{str(num_hidden_layers_in_network_doing_nas)}.py"


    raw_test_dataset = SustainBenchTextTorchDataset(
        data_dir=DATA_DIR,
        features=features,
        target=target,
        model_type=model_type,
        data_split="test",
        classification_threshold = classification_threshold
    )
    input_dim = raw_test_dataset.embeddings[0].shape[0]
    model = model_class(input_dim=input_dim, model_type=model_type,
                        num_hidden_layers=num_hidden_layers_in_network_doing_nas)


    train_dataset = serialize(SustainBenchTextTorchDataset, data_dir=DATA_DIR,
        features=features,
        target=target,
        model_type=model_type,
        data_split="train",
        classification_threshold=classification_threshold)
    test_dataset = serialize(SustainBenchTextTorchDataset, data_dir=DATA_DIR,
        features=features,
        target=target,
        model_type=model_type,
        data_split="test",
        classification_threshold = classification_threshold)
    trainer = None
    if model_type == model_type.classification:
        trainer = pl.Classification(train_dataloader=pl.DataLoader(train_dataset, batch_size=model_batch_size),
                                    val_dataloaders=pl.DataLoader(test_dataset, batch_size=model_batch_size),
                                    max_epochs=model_epochs)#, criterion=nn.BCELoss)
    elif model_type == model_type.regression:
        trainer = pl.Regression(train_dataloader=pl.DataLoader(train_dataset, batch_size=model_batch_size),
                                    val_dataloaders=pl.DataLoader(test_dataset, batch_size=model_batch_size),
                                    max_epochs=model_epochs)
    else:
        raise NotImplementedError



    simple_strategy = strategy.Random()

    exp = RetiariiExperiment(model, trainer, [], simple_strategy)

    exp_config = RetiariiExeConfig('local')
    exp_config.experiment_name = 'NAS_search_random_strategy_' + model_type.name
    exp_config.trial_concurrency = 2
    exp_config.max_trial_number = 20
    exp_config.training_service.use_active_gpu = False
    exp_config.execution_engine="base"
    export_formatter = 'code'

    # uncomment this for graph-based execution engine
    # exp_config.execution_engine = 'base'
    # export_formatter = 'code'

    final_models = ""
    exp.run(exp_config, 8081 + random.randint(0, 100))
    for model_code in exp.export_top_models(formatter=export_formatter):
        final_models = model_code

    writeToAnyFile(final_models, os.path.join("NAS_SELECTED_MODELS", outfile_name))