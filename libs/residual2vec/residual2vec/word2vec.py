# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-04-20 14:33:08
# @Last Modified by:   Sadamori Kojaku
# @Last Modified time: 2023-04-13 10:56:58
"""Embedding module for Residual2Vec.

This module is a modified version of the Word2Vec module in
https://github.com/theeluwin/pytorch-sgn
"""
from scipy import sparse
import numpy as np
import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from residual2vec.fastRP import fastRP


class Word2Vec(nn.Module):
    def __init__(self, vocab_size, embedding_size, padding_idx, device):
        super(Word2Vec, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.ivectors = nn.Embedding(
            self.vocab_size + 1, self.embedding_size, padding_idx=padding_idx
        )
        self.ovectors = nn.Embedding(
            self.vocab_size + 1, self.embedding_size, padding_idx=padding_idx
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


class Word2VecRP(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        padding_idx,
        device,
        net,
        rp_layer_size=1024,
        hidden_layer_size=128,
    ):
        super(Word2VecRP, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_layer_size = hidden_layer_size
        self.rp_layer_size = rp_layer_size
        self.ibase = nn.Embedding(
            self.vocab_size + 1, self.rp_layer_size, padding_idx=padding_idx
        )
        self.obase = nn.Embedding(
            self.vocab_size + 1, self.rp_layer_size, padding_idx=padding_idx
        )
        self.ivector = nn.Linear(self.rp_layer_size, self.hidden_layer_size)
        self.ovector = nn.Linear(self.rp_layer_size, self.hidden_layer_size)
        self.ivector_out = nn.Linear(self.hidden_layer_size, self.embedding_size)
        self.ovector_out = nn.Linear(self.hidden_layer_size, self.embedding_size)

        self.drop_out = nn.Dropout(p=0.5)
        self.relu = nn.ELU()

        # Initialize
        base = fastRP(net, self.rp_layer_size, 3, normalize=True)
        base = np.random.randn(self.vocab_size, self.rp_layer_size)
        base = np.einsum("ij,i->ij", base, 1 / np.linalg.norm(base, axis=1))
        base = torch.from_numpy(base).type(torch.FloatTensor)
        self.ibase.weight = nn.Parameter(
            torch.cat(
                [
                    base,
                    torch.zeros(1, self.rp_layer_size),
                ]
            )
        )
        self.obase.weight = nn.Parameter(
            torch.cat(
                [
                    base,
                    torch.zeros(1, self.rp_layer_size),
                ]
            )
        )
        self.ivector.weight = nn.Parameter(
            FloatTensor(self.rp_layer_size, self.hidden_layer_size)
            .uniform_(-0.5 / self.hidden_layer_size, 0.5 / self.hidden_layer_size)
            .T
        )
        self.ovector.weight = nn.Parameter(
            FloatTensor(self.rp_layer_size, self.hidden_layer_size)
            .uniform_(-0.5 / self.hidden_layer_size, 0.5 / self.hidden_layer_size)
            .T
        )
        self.ivector_out.weight = nn.Parameter(
            FloatTensor(self.hidden_layer_size, self.embedding_size)
            .uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
            .T
        )
        self.ovector_out.weight = nn.Parameter(
            FloatTensor(self.hidden_layer_size, self.embedding_size)
            .uniform_(-0.5 / self.embedding_size, 0.5 / self.embedding_size)
            .T
        )

        self.ibase.weight.requires_grad = False
        self.obase.weight.requires_grad = False
        self.ivector.weight.requires_grad = True
        self.ovector.weight.requires_grad = True
        self.ivector_out.weight.requires_grad = True
        self.ovector_out.weight.requires_grad = True
        self.to(self.device)

    def forward(self, data):
        return self.forward_i(data)

    def forward_i(self, data):
        # v = LongTensor(data).to(self.device)
        # v = v.cuda() if self.ivectors.weight.is_cuda else v
        x = self.ibase(data)
        x = self.ivector(x)
        x = self.relu(x)
        # x = self.drop_out(x)
        x = self.ivector_out(x)
        return x

    def forward_o(self, data):
        x = self.obase(data)
        x = self.ovector(x)
        x = self.relu(x)
        # x = self.drop_out(x)
        x = self.ovector_out(x)
        return x


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
