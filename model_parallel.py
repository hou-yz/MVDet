# -*- coding: utf-8 -*-
"""
Single-Machine Model Parallel Best Practices
================================
**Author**: `Shen Li <https://mrshenli.github.io/>`_
Model parallel is widely-used in distributed training
techniques. Previous posts have explained how to use
`DataParallel <https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html>`_
to train a neural network on multiple GPUs; this feature replicates the
same model to all GPUs, where each GPU consumes a different partition of the
input data. Although it can significantly accelerate the training process, it
does not work for some use cases where the model is too large to fit into a
single GPU. This post shows how to solve that problem by using **model parallel**,
which, in contrast to ``DataParallel``, splits a single model onto different GPUs,
rather than replicating the entire model on each GPU (to be concrete, say a model
``m`` contains 10 layers: when using ``DataParallel``, each GPU will have a
replica of each of these 10 layers, whereas when using model parallel on two GPUs,
each GPU could host 5 layers).
The high-level idea of model parallel is to place different sub-networks of a
model onto different devices, and implement the ``forward`` method accordingly
to move intermediate outputs across devices. As only part of a model operates
on any individual device, a set of devices can collectively serve a larger
model. In this post, we will not try to construct huge models and squeeze them
into a limited number of GPUs. Instead, this post focuses on showing the idea
of model parallel. It is up to the readers to apply the ideas to real-world
applications.
.. note::
    For distributed model parallel training where a model spans multiple
    servers, please refer to
    `Getting Started With Distributed RPC Framework <rpc_tutorial.html>`__
    for examples and details.
Basic Usage
-----------
"""

######################################################################
# Let us start with a toy model that contains two linear layers. To run this
# model on two GPUs, simply put each linear layer on a different GPU, and move
# inputs and intermediate outputs to match the layer devices accordingly.
#

import torch
import torch.nn as nn
import torch.optim as optim


class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.net1 = torch.nn.Linear(10, 10)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(10, 5)

    def forward(self, x):
        x = self.relu(self.net1.to('cuda:0')(x.to('cuda:0')))
        # x = x.to('cuda:1') + nn.functional.sigmoid(self.net1.to('cuda:1')(x.to('cuda:1')))
        return self.net2.to('cuda:1')(x.to('cuda:1'))


pass

######################################################################
# Note that, the above ``ToyModel`` looks very similar to how one would
# implement it on a single GPU, except the five ``to(device)`` calls which
# place linear layers and tensors on proper devices. That is the only place in
# the model that requires changes. The ``backward()`` and ``torch.optim`` will
# automatically take care of gradients as if the model is on one GPU. You only
# need to make sure that the labels are on the same device as the outputs when
# calling the loss function.

model = ToyModel()
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

optimizer.zero_grad()
outputs = model(torch.randn(20, 10))
labels = torch.randn(20, 5).to('cuda:1')
loss_fn(outputs, labels).backward()
optimizer.step()
