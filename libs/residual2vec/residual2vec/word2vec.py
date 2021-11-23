"""Embedding module for Residual2Vec.

This module is a modified version of the Word2Vec module in
https://github.com/theeluwin/pytorch-sgn
"""
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx):
        super(Word2Vec, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ovectors = nn.Embedding(
            self.vocab_size, self.embedding_size, padding_idx=padding_idx
        )
        self.ivectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    FloatTensor(self.vocab_size, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ovectors.weight = nn.Parameter(
            torch.cat(
                [
                    torch.zeros(1, self.embedding_size),
                    FloatTensor(self.vocab_size, self.embedding_size).uniform_(
                        -0.5 / self.embedding_size, 0.5 / self.embedding_size
                    ),
                ]
            )
        )
        self.ivectors.weight.requires_grad = True
        self.ovectors.weight.requires_grad = True

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(v)

    def forward_o(self, data):
        v = LongTensor(data)
        v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(v)


class NegativeSampling(nn.Module):
    def __init__(self, embedding):
        super(NegativeSampling, self).__init__()
        self.embedding = embedding
        self.weights = None

    def forward(self, iword, owords, nwords):
        ivectors = self.embedding.forward_i(iword).unsqueeze(2)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords).neg()
        logsigmoid = nn.LogSigmoid()
        oloss = (
            torch.bmm(ovectors, ivectors)
            .squeeze()
            .sigmoid()
            .clamp(1e-12, 1)
            .log()
            .mean(1)
        )
        nloss = (
            torch.bmm(nvectors, ivectors)
            .squeeze()
            .sigmoid()
            .clamp(1e-12, 1)
            .log()
            .mean(1)
        )
        return -(oloss + nloss).mean()
