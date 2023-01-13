import torch
import sys
import os
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

from unimodals.common_models import GRU, MLP, Sequential, Identity  # noqa
from training_structures.ensemble import train, test  # noqa
from datasets.affect.get_data import get_dataloader  # noqa
from fusions.ensemble_fusions import AdditiveEnsemble  # noqa

traindata, validdata, testdata = get_dataloader(
    'data_files/mosi_raw.pkl', robust_test=False, max_pad=True, data_type='mosi', max_seq_len=50)

modalities = [0,1,2]
dataset = 'mosi'

# mosi/mosei
if dataset == 'mosi':
    d_v = (35, 600)
    d_a = (74, 600)
    d_l = (300, 600)
elif dataset == 'mosei':
    d_v = (713, 70)
    d_a = (74, 200)
    d_l = (300, 600)

# humor/sarcasm
elif dataset == 'humor' or dataset == 'sarcasm':
    d_v = (371, 70)
    d_a = (81, 200)
    d_l = (300, 600)
    
config = [d_v, d_a, d_l]
d_modalities = [config[i] for i in modalities]

# train all unimodal models first
encoders = [torch.load(f'mosiencoder{i}.pt') for i in modalities]
heads = [torch.load(f'mosihead{i}.pt') for i in modalities]
ensemble = AdditiveEnsemble().cuda()


train(encoders, heads, ensemble, traindata, validdata, 200, task="regression", optimtype=torch.optim.AdamW, lr=2e-3,
      weight_decay=0.01, criterion=torch.nn.L1Loss(), save_model='ensemble.pt', modalities=modalities)


print("Testing:")
ensemble = torch.load('ensemble.pt').cuda()
test(ensemble, testdata, 'affect', criterion=torch.nn.L1Loss(),
     task="posneg-classification", no_robust=True)
