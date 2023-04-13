# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-04-20 14:33:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-07 11:30:33
"""Embedding module for Residual2Vec.

This module is a modified version of the Word2Vec module in
https://github.com/theeluwin/pytorch-sgn
"""
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, device):
        super(Word2Vec, self).__init__()
        self.device = device
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
        self.to(self.device)

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        # v = LongTensor(data).to(self.device)
        # v = v.cuda() if self.ivectors.weight.is_cuda else v
        return self.ivectors(data)

    def forward_o(self, data):
        # v = LongTensor(data).to(self.device)
        # v = v.cuda() if self.ovectors.weight.is_cuda else v
        return self.ovectors(data)


class NegativeSampling(nn.Module):
    def __init__(self, embedding):
        super(NegativeSampling, self).__init__()
        self.embedding = embedding
        self.weights = None
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, iword, owords, nwords, niwords=None):
        """Loss

        :param iword: center nodes
        :type iword: torch.Tensor
        :param owords: context nodes
        :type owords: torch.Tensor
        :param nwords: random context nodes
        :type nwords: torch.Tensor
        :param niwords: random center nodes, defaults to None. If set to None, the center nodes will be used for computing the negative loss.
        :type niwords: torch.Tensor, optional
        :return: _description_
        :rtype: _type_
        """

        ivectors = self.embedding.forward_i(iword)
        ovectors = self.embedding.forward_o(owords)
        nvectors = self.embedding.forward_o(nwords)
        oloss = self.logsigmoid((ovectors * ivectors).sum(dim=1))
        if niwords is None:
            nloss = self.logsigmoid((nvectors * ivectors).sum(dim=1).neg())
        elif niwords is not None:
            nivectors = self.embedding.forward_i(niwords)
            nloss = self.logsigmoid((nvectors * nivectors).sum(dim=1).neg())
        return -(oloss + nloss).mean()
