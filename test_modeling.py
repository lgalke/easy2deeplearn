import torch
import torch.nn as nn
from modeling import TiedLinear, TiedEmbedding

def test_TiedLinear_type():
    f = nn.Linear(20,10)
    g = TiedLinear(f)

    assert isinstance(f, nn.Linear)
    assert isinstance(g, nn.Linear)

def test_TiedLinear_shapes():

    orig = nn.Linear(20,10)

    tied = TiedLinear(orig)

    x = torch.randn(20)

    z = orig(x)

    x_rec = tied(z)

    x = torch.randn(5, 20)

    z = orig(x)

    x_rec = tied(z)

def test_TiedLinear_intervention():

    orig = nn.Linear(20,10)

    tied = TiedLinear(orig)


    orig.weight.data[2, 7] = 42.0
    assert tied.weight[2, 7] == 42.0

def test_TiedLinear_weight_init():
    f = nn.Linear(20,10)
    g = TiedLinear(f)
    f.reset_parameters()
    assert (g.weight.data == f.weight.data).all()

def test_TiedLinear_bias_init():
    f = nn.Linear(20,10)
    g = TiedLinear(f)
    tmp = f.weight.data.clone()
    nn.init.ones_(g.bias) # temporarily set to ones
    g.reset_parameters()  # should reset to all zerosk
    assert (g.bias.data == 0.0).all()
    # f.weight should be untouched by reset parameters
    assert (f.weight.data == tmp).all()


def test_TiedEmbedding_type():
    f = nn.Linear(2,3)
    g = TiedEmbedding(f)

    assert isinstance(f, nn.Linear)
    assert isinstance(g, nn.Embedding)

def test_TiedEmbedding_shapes():
    f = nn.Linear(20,30)
    g = TiedEmbedding(f)
    x = torch.randn(20).view(1,-1)

    # Test that it doesnt mess with orig Linear
    h = f(x)
    assert h.size(1) == 30

    x = torch.tensor([0,1])
    x_emb = g(x)
    # assert x_emb.size(0) == 2  # same as input long vals
    assert x_emb.size(-1) == 30  # same as linear's output dim


def test_TiedEmbedding_values():
    f = nn.Linear(20,30)
    g = TiedEmbedding(f)
    x = torch.randn(20).view(1,-1)

    # Test that it doesnt mess with orig Linear
    h = f(x)
    assert h.size(1) == 30

    x = torch.tensor([0,1,19])
    x_emb = g(x)
    # assert x_emb.size(0) == 2  # same as input long vals
    x_emb[0] = f.weight.T[0]
    x_emb[1] = f.weight.T[1]
    x_emb[2] = f.weight.T[19]
