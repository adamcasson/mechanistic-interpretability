import random
from typing import Sequence

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader, Dataset


class AdditionDataset(Dataset):
    """
    Code source: https://github.com/karpathy/minGPT/blob/master/projects/adder/adder.py

    Creates n-digit addition problems. For example, if n=2, then an example
    addition problem would be to add 85 + 50 = 135. This problem would be
    represented as the following string for the GPT:
    "8550531"
    This is because:
    - we are discarding the + and =, which are not necessary. We just encode the digits
      of the input numbers concatenated together.
    - the result 135 is encoded backwards to make the addition easier to learn for the
      GPT model, because of how the addition algorithm works.
    As one more example, the problem 6 + 39 = 45 would be encoded as:
    "0639054"
    where you will notice that we are padding with zeros to make sure that we always
    produce strings of the exact same size: n + n + (n + 1). When n=2, this is 7.
    At test time, we will feed in an addition problem by giving the first 2n digits,
    and hoping that the GPT model completes the sequence with the next (n+1) digits
    correctly.
    """

    def __init__(self, n_digit: int, idx: Sequence[int], reverse_sum: bool = False) -> None:
        self.n_digit = n_digit
        self.idx = idx
        self.reverse_sum = reverse_sum

    def get_vocab_size(self):
        return 10  # digits 0..9

    def get_block_size(self):
        # a,b,a+b, and +1 due to potential carry overflow,
        # but then also -1 because very last digit doesn't ever plug back
        # as there is no explicit <EOS> token to predict, it is implied
        return 3 * self.n_digit + 1 - 1

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, idx):
        # given a problem index idx, first recover the associated a + b
        idx = self.idx[idx]
        nd = 10**self.n_digit
        a = idx // nd
        b = idx % nd
        # calculate the "label" of the addition problem a + b
        c = a + b
        # encode the digits of a, b, c into strings
        astr = f'%0{self.n_digit}d' % a
        bstr = f'%0{self.n_digit}d' % b
        cstr = f'%0{self.n_digit+1}d' % c  # reverse c to make addition easier
        if self.reverse_sum:
            cstr = cstr[::-1]
        render = astr + bstr + cstr
        dix = [int(s) for s in render]  # convert each character to its token index
        # x will be input to GPT and y will be the associated expected outputs
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)  # predict the next token in the sequence
        # we will only train in the output locations. -1 will mask loss to zero
        y[: self.n_digit * 2 - 1] = -1
        return x, y


class DigitAdditionDataModule(LightningDataModule):
    def __init__(
        self,
        n_digit: int,
        frac_train: float = 0.8,
        reverse_sum: bool = False,
        train_batch_size: int = 256,
        val_batch_size: int = 256,
        pred_batch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__()
        # split up all addition problems into either training data or test data
        assert (
            n_digit <= 3
        ), "the lines below would be very memory inefficient, in future maybe refactor to support"
        num = (
            10**n_digit
        ) ** 2  # total number of possible addition problems with ndigit numbers
        idx_all = [i for i in range(num)]
        random.Random(seed).shuffle(idx_all)
        div = int(frac_train * len(idx_all))
        train_idx = idx_all[:div]
        val_idx = idx_all[div:]

        self.train_dataset = AdditionDataset(
            n_digit=n_digit, idx=train_idx, reverse_sum=reverse_sum
        )
        self.val_dataset = AdditionDataset(n_digit=n_digit, idx=val_idx, reverse_sum=reverse_sum)
        self.pred_dataset = AdditionDataset(n_digit=n_digit, idx=idx_all, reverse_sum=reverse_sum)
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.pred_batch_size = pred_batch_size

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.pred_dataset, batch_size=self.pred_batch_size)
