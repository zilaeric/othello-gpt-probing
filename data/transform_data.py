from othello import get as get_othello

import itertools
import torch
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, data):
        if hasattr(data, "ood_perc"):
            ood_perc = data.ood_perc
            data.ood_perc = 0  # shut down the randomness
        chars = sorted(list(set(list(itertools.chain.from_iterable(data)))) + [-100, ])
        data_size, vocab_size = len(data), len(chars)  # vocab size 61, with -100 sorted to the front
        max_len = max([len(data[_]) for _ in range(len(data))])  # should be 60 in Othello
        print('Dataset created has %d sequences, %d unique words.' % (data_size, vocab_size))
        
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.max_len = max_len
        self.block_size = max_len - 1  # for autoregressive training
        self.vocab_size = vocab_size
        if hasattr(data, "ood_perc"):
            data.ood_perc = ood_perc  # turn on the randomness
        self.data = data
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx]
        if len(chunk) != self.max_len:
            chunk += [-100, ] * (self.max_len - len(chunk))  # -100 can be ignored in CE
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        """
        arrange data and targets so that the first i elements of x
        will be asked to predict the i-th element of y. Notice that
        the eventual language model will actually make block_size
        individual predictions at the same time based on this data,
        so we are being clever and amortizing the cost of the forward
        pass of the network. So for example if block_size is 4, then
        we could e.g. sample a chunk of text "hello", the integers in
        x will correspond to "hell" and in y will be "ello". This will
        then actually "multitask" 4 separate examples at the same time
        in the language model:
        - given just "h", please predict "e" as next
        - given "he" please predict "l" next
        - given "hel" predict "l" next
        - given "hell" predict "o" next
        
        In addition, because the DataLoader will create batches of examples,
        every forward/backward pass during traning will simultaneously train
        a LOT of predictions, amortizing a lot of computation. In particular,
        for a batched input of integers X (B, T) where B is batch size and
        T is block_size and Y (B, T), the network will during training be
        simultaneously training to make B*T predictions, all at once! Of course,
        at test time we can paralellize across batch B, but unlike during training
        we cannot parallelize across the time dimension T - we have to run
        a forward pass of the network to recover the next single character of the 
        sequence along each batch dimension, and repeatedly always feed in a next
        character to get the next one.
        
        So yes there is a big asymmetry between train/test time of autoregressive
        models. During training we can go B*T at a time with every forward pass,
        but during test time we can only go B at a time, T times, with T forward 
        passes.
        """
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y

print("Making dataset")
othello = get_othello(ood_num=-1, data_root=None, wthor=True)
train_dataset = CharDataset(othello)

full_seqs = list(filter(lambda x: len(x)==60, train_dataset.data.sequences))
print(len(full_seqs))
board_seqs = torch.tensor(full_seqs)
print(board_seqs.numel())

# %%
# n = 50000
# board_seqs = torch.zeros((n, 60), dtype=int)
# for c, seq in enumerate(tqdm(othello.sequences)):
#     board_seqs[c, :len(seq)] = torch.tensor(seq)
#     if c == n-1:
#         break
# %%

board_seqs_string = board_seqs
print(board_seqs_string.numel())

board_seqs_int = board_seqs_string.clone()
board_seqs_int[board_seqs_string < 29] += 1
board_seqs_int[(board_seqs_string >= 29) & (board_seqs_string <= 34)] -= 1
board_seqs_int[(board_seqs_string > 34)] -= 3
rand = torch.randint(0, 1000000, (20,))
print(board_seqs_int.flatten()[rand])
print(board_seqs_string.flatten()[rand])

indices = torch.randperm(len(board_seqs_int))
board_seqs_int = board_seqs_int[indices]
board_seqs_string = board_seqs_string[indices]
torch.save(board_seqs_int, "mechanistic_interpretability/board_seqs_int.pth")
torch.save(board_seqs_string, "mechanistic_interpretability/board_seqs_string.pth")

board_seqs_int = torch.load("mechanistic_interpretability/board_seqs_int.pth")
board_seqs_string = torch.load("mechanistic_interpretability/board_seqs_string.pth")
print(board_seqs_int.shape)