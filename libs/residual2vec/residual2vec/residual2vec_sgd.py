"""Residual2Vec module."""
import numpy as np
from numba import njit
from scipy import sparse
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from residual2vec import utils
from residual2vec.random_walk_sampler import RandomWalkSampler
from residual2vec.word2vec import NegativeSampling, Word2Vec


class residual2vec_sgd:
    """Residual2Vec based on the stochastic gradient descent."""

    def __init__(
        self,
        noise_sampler,
        window_length=10,
        batch_size=4,
        num_walks=100,
        walk_length=40,
        p=1,
        q=1,
        cuda=False,
    ):
        self.window_length = window_length
        self.sampler = noise_sampler
        self.cuda = cuda
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.p = p
        self.q = q
        self.batch_size = batch_size

    def fit(self, A):
        # Set up the graph object for efficient sampling
        self.A = A
        self.n_nodes = A.shape[0]
        self.sampler.fit(A)

    def transform(self, dim):

        # Set up the embedding model
        PADDING_IDX = self.n_nodes - 1
        model = Word2Vec(
            vocab_size=self.n_nodes + 1, embedding_size=dim, padding_idx=PADDING_IDX
        )
        neg_sampling = NegativeSampling(embedding=model)
        if self.cuda:
            model = model.cuda()

        # Set up the Training dataset
        dataset = TripletDataset(
            A=self.A,
            num_walks=self.num_walks,
            window_length=self.window_length,
            noise_sampler=self.sampler,
            padding_id=PADDING_IDX,
            walk_length=self.walk_length,
            p=self.p,
            q=self.q,
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Training
        optim = Adam(model.parameters())
        pbar = tqdm(dataloader)
        for iword, owords, nwords in pbar:
            loss = neg_sampling(iword, owords, nwords)
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss=loss.item())

        self.in_vec = model.ivectors.weight.data.cpu().numpy()[: PADDING_IDX + 1, :]
        self.out_vec = model.ovectors.weight.data.cpu().numpy()[: PADDING_IDX + 1, :]
        return self.in_vec


class TripletDataset(Dataset):
    def __init__(
        self,
        A,
        num_walks,
        window_length,
        noise_sampler,
        padding_id,
        walk_length=40,
        p=1,
        q=1,
    ):
        self.A = A
        self.num_walks = num_walks
        self.window_length = window_length
        self.noise_sampler = noise_sampler
        self.walk_length = walk_length
        self.padding_id = padding_id
        self.rw_sampler = RandomWalkSampler(A, walk_length=walk_length, p=p, q=q)
        self.node_order = np.random.choice(A.shape[0], A.shape[0], replace=False)
        self.n_nodes = A.shape[0]

        # Counter and Memory
        self.rw_id = 0
        self.t_walk = 0  #
        self.walk = None

        # Initialize
        self._generate_samples()

    def __len__(self):
        return self.num_walks * self.walk_length

    def __getitem__(self, idx):
        if self.t_walk == len(self.walk):
            self._generate_samples()

        center = self.walk[self.t_walk]
        context = _get_context(
            self.walk,
            len(self.walk),
            self.t_walk,
            self.window_length,
            padding_id=self.padding_id,
        )
        random_context = self.noise_sampler.sampling(center, len(context))
        self.t_walk += 1
        return center, context.astype(np.int64), random_context.astype(np.int64)

    def _generate_samples(self):
        walk = self.rw_sampler.sampling(self.node_order[self.rw_id])
        self.rw_id = (self.rw_id + 1) % self.n_nodes
        self.walk = walk
        self.t_walk = 0


@njit(nogil=True)
def _get_context(walk, walk_len, t_walk, window_length, padding_id):
    retval = padding_id * np.ones(2 * window_length, dtype=np.int64)
    for i in range(window_length):
        if t_walk - 1 - i < 0:
            break
        retval[window_length - 1 - i] = walk[t_walk - 1 - i]

    for i in range(window_length):
        if t_walk + 1 + i >= walk_len:
            break
        retval[window_length + i] = walk[t_walk + 1 + i]
    return retval
