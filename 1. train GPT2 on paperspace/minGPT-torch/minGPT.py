from pytorch_lightning import seed_everything
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torch.utils.data import Dataset, DataLoader
from mingpt.model import GPT
from pytorch_lightning import Trainer
from mingpt.lr_decay import LearningRateDecayCallback


seed_everything(42)
block_size = 128 # spatial extent of the model for its context

class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = list(set(data))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return math.ceil(len(self.data) / (self.block_size + 1))

    def __getitem__(self, idx):
        # we're actually going to "cheat" and pick a spot in the dataset at random
        i = np.random.randint(0, len(self.data) - (self.block_size + 1))
        chunk = self.data[i:i+self.block_size+1]
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y
    
text = open('minGPT-torch/input.txt', 'r').read() # don't worry we won't run out of file handles
train_dataset = CharDataset(text, block_size) # one line of poem is roughly 50 characters
train_loader = DataLoader(train_dataset, batch_size=64)

model = GPT(vocab_size=train_dataset.vocab_size, 
            block_size=train_dataset.block_size,
            n_layer=8, 
            n_head=8, 
            n_embd=512, 
            learning_rate=6e-4)


# scheduler
lr_decay = LearningRateDecayCallback(learning_rate=6e-4, warmup_tokens=512*20,
                                    final_tokens=00*len(train_dataset)*block_size)

trainer = Trainer(gpus=1, distributed_backend='ddp',
                    amp_backend='native',
                    precision=16, max_epochs=1,
                    gradient_clip_val=1.0, 
                    callbacks=[lr_decay], 
                    progress_bar_refresh_rate=1, 
                    row_log_interval=1)
trainer.fit(model, train_loader)

from mingpt.utils import sample

context = "O God, I code but"
x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None,...].to(model.device)
y = sample(model, x, 1000, temperature=0.9, sample=True, top_k=5)[0]
completion = ''.join([train_dataset.itos[int(i)] for i in y])
print(completion)