import os
import numpy as np
from torch.utils.data import Dataset
import string
from tqdm import tqdm
import logging
from utils import load, g2p, phone2seq

class SongsDataset(Dataset):
    def __init__(self, dataset, sr=22050, dummy=False):
        '''
        :param dataset:     a list of song with line level annotation
        :param sr:          sampling rate
        '''

        super(SongsDataset, self).__init__()
        self.dataset = dataset
        self.sr = sr

        txt = open(f"data/{dataset}.txt", "r").read()
        self.filenames = txt.split("\n")
        self.length = len(self.filenames)

    def __getitem__(self, index):
        filename = self.filenames[index]
        audio, _ = load(os.path.join("data", f"{filename}.mp3"), sr=self.sr, mono=True)
        txt = open(f"data/{filename}.txt", "r").read()
        phone_seq = phone2seq(g2p(txt))
        return audio, None, None, phone_seq, None

    def __len__(self):
        return self.length
