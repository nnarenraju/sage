import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset

# Packages required to read data
import os
import re
import h5py
import glob
import random
import numpy as np
import pandas as pd

from pycbc.psd import inverse_spectrum_truncation
from pycbc.types import TimeSeries

## Options
whitening = True
highpass = True


""" Get training and testing data """
# Training and testing data
lookup_table = './extlinks.hdf'
with h5py.File(lookup_table, 'a') as fp:
    ids = np.array(fp['id'][:])
    paths = np.array([foo.decode('utf-8') for foo in fp['path']])
    targets = np.array(fp['target'][:])
    dstype = np.array([foo.decode('utf-8') for foo in fp['type']])

lookup = list(zip(ids, paths, targets, dstype))
train = pd.DataFrame(lookup, columns=['id', 'path', 'target', 'dstype'])

training_data = train.loc[train['dstype'] == 'training']
validation_data = train.loc[train['dstype'] == 'validation']
# Get the required number of samples from each
train_idx, val_idx = (training_data['id'].values, validation_data['id'].values)
train_fold = train.iloc[train_idx]
valid_fold = train.iloc[val_idx]

# Define a dataset object
class MLMDC1(Dataset):
    
    def __init__(self, data_paths, targets):
        super().__init__()
        self.data_paths = data_paths
        self.targets = targets

        # Set CUDA device for pin_memory if needed
        if bool(re.search('cuda', self.cfg.store_device)):
            setattr(self, 'foo', torch.cuda.set_device(self.cfg.store_device))

        # Random Noise realisation
        self.noise_idx = np.argwhere(self.targets == 0).flatten()
        
        """ Keep ExternalLink Lookup table open till end of run """
        lookup = './extlinks.hdf'
        self.extmain = h5py.File(lookup, 'r', libver='latest')
        
    def __len__(self):
        return len(self.data_paths)
    
    def _read_(self, data_path):
        # Store all params within chunk file
        targets = {}
        # Get data from ExternalLink'ed lookup file
        HDF5_Dataset, didx = os.path.split(data_path)
        # Dataset Index should be an integer
        didx = int(didx)
        # Check whether data is signal or noise with target
        target = 1 if bool(re.search('signal', HDF5_Dataset)) else 0
        # Access group
        group = self.extmain[HDF5_Dataset]
        
        if not target:
            ## Read noise data
            noise_1 = np.array(group['noise_1'][didx])
            noise_2 = np.array(group['noise_2'][didx])
            sample = np.stack([noise_1, noise_2], axis=0)
        else:
            ## Read signal data
            h_plus = np.array(group['h_plus'][didx])
            h_cross = np.array(group['h_cross'][didx])
            sample = np.stack([h_plus, h_cross], axis=0)
        
        return (sample, target)
    
    def _noise_realisation_(self, sample, targets, params):
        """ Finding random noise realisation for signal """
        if targets['gw']:
            # Pick a random noise realisation to add to the signal
            random_noise_idx = random.choice(self.noise_idx)
            random_noise_data_path = self.data_paths[random_noise_idx]
            # Read the noise data
            pure_noise, _ = self._read_(random_noise_data_path)
            noisy_signal = sample + pure_noise
        else:
            # If the sample is pure noise
            noisy_signal = sample
        
        return noisy_signal
    
    def __getitem__(self, idx):
        # Setting the unique seed for given sample
        np.random.seed(idx+1)
        # Get data paths for external link
        data_path = self.data_paths[idx]
        ## Read the sample(s)
        pure_sample, target = self._read_(data_path)
        ## Add noise realisation to the signals
        noisy_sample = self._noise_realisation_(pure_sample, target)
        # Transformation (Whitening)
        if whitening:
            psds = {}
            psd_files = glob.glob("/local/scratch/igr/nnarenraju/dataset_simple/psds/*")
            for psd_file in psd_files:
                with h5py.File(psd_file, 'r') as fp:
                    data = np.array(fp['data'])
                    delta_f = fp.attrs['delta_f']
                    name = fp.attrs['name']
                    
                psd_data = FrequencySeries(data, delta_f=delta_f)
                # Store PSD data into lookup dict
                psds[name] = psd_data
            psds_data = [psds['median_det1'], psds['median_det2']]
            delta_f = None
            max_filter_len = int(round(5.0 * 2048.))
            sample = []
            for psd, det_data in zip(psds_data, sample):
                signal = TimeSeries(det_data, delta_t=1./2048.)
                psd = inverse_spectrum_truncation(psd,
                                        max_filter_len=max_filter_len,
                                        low_frequency_cutoff=15.0,
                                        trunc_method='hann')
                white = (signal.to_frequencyseries(delta_f=delta_f) / psd**0.5).to_timeseries()
                white = white[int(max_filter_len/2):int(len(white)-max_filter_len/2)]
                sample.append(white)
            sample = np.stack(sample, axis=0)

        # Reducing memory footprint
        _sample = np.array(sample, dtype=np.float32)
        sample = _sample[:].copy()
        sample = torch.from_numpy(sample)
        # Transformation (HighPass)
        sample = [torchaudio.functional.highpass_biquad(dat, 2048, 20.0) for dat in sample]
        sample = torch.stack(sample, dim=0)
        return sample, target

train_dataset = MLMDC1(
        data_paths=train_fold['path'].values, targets=train_fold['target'].values)
        
valid_dataset = MLMDC1(
        data_paths=valid_fold['path'].values, targets=valid_fold['target'].values)

# DataLoader Params
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

for X, y in train_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

raise






# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)

# Optimising the model parameters
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# Training
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")


