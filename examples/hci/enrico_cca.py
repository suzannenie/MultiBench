from unimodals.common_models import VGG16, VGG16Slim, DAN, Linear, MLP, VGG11Slim, VGG11Pruned
import torch
from memory_profiler import memory_usage
from private_test_scripts.all_in_one import all_in_one_train, all_in_one_test
from datasets.enrico.get_data import get_dataloader
from fusions.common_fusions import Concat
from objective_functions.objectives_for_supervised_learning import CCA_objective
from training_structures.Supervised_Learning import train, test
import sys
import os
from torch import nn
sys.path.append(os.getcwd())


dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, testdata = dls
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights)).cuda()
# encoders=[VGG16Slim(64).cuda(), DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
# head = Linear(96, 20)
encoders = [VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda(
), VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()]
# encoders = [DAN(4, 16, dropout=True, dropoutp=0.25).cuda(), DAN(28, 16, dropout=True, dropoutp=0.25).cuda()]
head = Linear(32, 20).cuda()

fusion = Concat().cuda()

allmodules = encoders + [head, fusion]


def trainprocess():
    train(encoders, fusion, head, traindata, validdata, 50, optimtype=torch.optim.Adam,
          lr=0.0001, weight_decay=0, objective=CCA_objective(16), objective_args_dict={})


all_in_one_train(trainprocess, allmodules)

print("Testing:")
model = torch.load('best.pt').cuda()

test(model, testdata, dataset='enrico')
