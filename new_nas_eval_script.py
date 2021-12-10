import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import r2_score, classification_report, confusion_matrix, roc_auc_score, accuracy_score
from experiments.experiment import ModelType

from utils.get_data_loader import *
from utils.file_utils import *

import numpy as np
import itertools
from importlib import import_module

import pdb


import time
from models.feedforward_network_with_nas import FeedforwardNetworkModuleForNAS, FeedforwardNetworkForNASModelInterface



CLASSIFICATION_THRESHOLD_DICT = {'asset_index': 0, 'sanitation_index': 3, 'water_index': 3, 'women_edu': 5}

TARGETS = ['asset_index', 'sanitation_index', 'water_index', 'women_edu']


#features = ['target_sentence', 'all_sentence', 'document']
features = ['target_sentence', 'document']





def get_data_loader(features, target, rebalance=True):
    ds = SustainBenchTextDataset(
        data_dir=PATH_TO_DATA_DIR,
        features=features,
        target=target,
        model_type='classification',
        classification_threshold=CLASSIFICATION_THRESHOLD_DICT[target],
        rebalance=rebalance
    )

    X_train, y_train = ds.get_data('train')
    X_val = X_train[int(0.9*X_train.shape[0]):]
    y_val = y_train[int(0.9*y_train.shape[0]):]
    X_train = X_train[:int(0.9 * X_train.shape[0])]
    y_train = y_train[:int(0.9*y_train.shape[0])]
    X_test, y_test = ds.get_data('test')

    #pdb.set_trace()

    print(f'train data shape: {X_train.shape}, test data shape: {X_test.shape}')
    print(
        f'{sum(y_train)} ({np.round(sum(y_train)/X_train.shape[0] * 100, 2)}%) positive examples in train data, {sum(y_test)} ({np.round(sum(y_test)/X_test.shape[0] * 100, 2)}%) positive examples in test data\n')

    X_train = torch.Tensor(X_train)
    y_train = torch.from_numpy(y_train)

    X_val = torch.Tensor(X_val)
    y_val = torch.from_numpy(y_val)

    X_test = torch.Tensor(X_test)
    y_test = torch.from_numpy(y_test)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset)

    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset)

    return train_loader, val_loader, test_loader

def train(train_loader, val_loader, net, learning_rate, num_epochs=10, logging=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    accuracy_values = list()
    for epoch in range(num_epochs):
        for i, (embeddings, labels) in enumerate(train_loader):
            embeddings = Variable(embeddings)
            labels = Variable(labels)

            optimizer.zero_grad()
            outputs = net(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if logging and (i+1) % 100 == 0:
                print(f'epoch: {epoch+1}/{num_epochs}, step: {int((i+1) / 100)}/{len(train_dataset)//batch_size}, loss: {loss.data}')

        total_preds = np.zeros(len(val_loader))
        y_labels = np.zeros(len(val_loader))
        for i, (embeddings,labels) in enumerate(val_loader):
            y_labels[i * 32:(i + 1) * 32] = labels

            embeddings = Variable(embeddings)
            labels = Variable(labels)

            outputs = net(embeddings)
            preds = np.argmax(outputs.detach().numpy(), axis=1)
            total_preds[i*32:(i+1)*32] = preds


        correct = (total_preds == y_labels).sum()
        accuracy_values.append(correct / len(val_loader))

    print(accuracy_values)
    return net, accuracy_values


def evaluate(test_loader, net, time_taken):
    y_true = []
    y_pred = []
    for i, (embeddings, labels) in enumerate(test_loader):
        embeddings = Variable(embeddings)
        outputs = net(embeddings)
        _, predicted = torch.max(outputs.data, 1)
        y_true.append(labels.item())
        y_pred.append(predicted.item())

    score = roc_auc_score(y_true, y_pred)
    print(f'basic neural network classifier fit time: {round(time_taken, 3)}s, roc auc: {round(score, 3)}')
    print(classification_report(y_true, y_pred))
    return score

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s)+1))

if __name__ == '__main__':
    FEATURE_INPUT_SIZE_DICT = {'target_sentence': 384, 'all_sentence': 384, 'document': 300}
    feature_combos = [list(combo) for combo in powerset(features) if len(combo) > 0]
    scores = np.zeros((len(TARGETS), len(feature_combos)))
    graph_data = [[list() for i in range(len(TARGETS))] for j in range(len(feature_combos))]

    model_type = ModelType.classification


    for i,target in enumerate(TARGETS):
        for j,feature_combo in enumerate(feature_combos):
            input_size = sum([FEATURE_INPUT_SIZE_DICT[feature] for feature in feature_combo])
            #hidden_size = int(input_size / 2)
            train_loader, val_loader, test_loader = get_data_loader(feature_combo, target, rebalance=True)
            features_string = ",".join(feature_combo)
            model_file_name = f"{model_type.name}_{features_string}_{target}_4"

            # import the correct module
            # pdb.set_trace()
            ModelModule = getattr(
                import_module(f'{PATH_TO_NAS_MODELS}.{model_file_name}'), '_model'
            )
            net = ModelModule()
#                model = FeedforwardNetworkForNASModelInterface(ModelModule, model_type = ModelType.classification)

            #net = Net(input_size, hidden_size, num_classes)

            start = time.time()
            net, acc_vals = train(train_loader, val_loader, net, 0.001)
            graph_data[i][j] = acc_vals
            end = time.time()
            curr_score = evaluate(test_loader, net, end - start)
            scores[i, j] = curr_score
            print("CURRENT SCORE: {}".format(curr_score))

        writeToJsonFile(scores.tolist(), "AUC_ROC_SCORES_NAS.json")
        writeToJsonFile(graph_data, "ACC_VALS_GRAPH_DATA.json")

