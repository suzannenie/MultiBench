import sys
import os
from torch import nn
sys.path.append(os.getcwd())
from training_structures.unimodal import train, test
from fusions.common_fusions import Concat
from datasets.enrico.get_data import get_dataloader
from unimodals.common_models import VGG16, VGG16Slim,DAN,Linear,MLP, VGG11Slim, VGG11Pruned, VGG16Pruned
from datasets.enrico.get_data_robust import get_dataloader_robust
from robustness.all_in_one import general_train, general_test
from memory_profiler import memory_usage

import torch

dls, weights = get_dataloader('datasets/enrico/dataset')
traindata, validdata, _ = dls
robustdata = get_dataloader_robust('datasets/enrico/dataset', wireframe_noise=False)
modalnum = 0
encoder=VGG11Slim(16, dropout=True, dropoutp=0.2, freeze_features=True).cuda()
head = Linear(16, 20).cuda()
# head = MLP(16, 32, 20, dropout=False).cuda()

allmodules = [encoder, head]

def trainprocess(filename_encoder, filename_head):
    train(encoder,head,traindata,validdata,50,optimtype=torch.optim.Adam,lr=0.0001,weight_decay=0,modalnum=modalnum,save_encoder=filename_encoder,save_head=filename_head)

filename = general_train(trainprocess, 'enrico_unimodal_0', encoder=True)

def testprocess(encoder, head, testdata):
    return test(encoder, head, testdata, modalnum=modalnum)

general_test(testprocess, filename, robustdata, encoder=True)
