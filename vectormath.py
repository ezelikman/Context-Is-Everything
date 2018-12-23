from scipy.special import erf
import scipy
import numpy as np
from numpy import exp
from numpy import tanh
from sklearn.preprocessing import normalize

# Returns arbitrary sigmoid
horizontal, vertical = 4.7, 5.1  # Original for erf

def sigmoid_exp(x, horizontal = horizontal, vertical = vertical):
    return (1/(1+exp((-x+0.5)*(horizontal*2.0)))-0.5)/(vertical/4.0) + 0.5  # proper


def sigmoid_tanh(x, horizontal = horizontal, vertical = vertical):
    return (tanh((x-0.5)*horizontal))/(2.0*vertical) + 0.5            # tanh


def sigmoid_erf(x, horizontal = horizontal, vertical = vertical):
    return 0.5+erf((x-0.5)*horizontal)/vertical                 # erf


def sigmoid(x, horizontal = horizontal, vertical = vertical):
    return sigmoid_exp(x, horizontal, vertical)

def sign_root(x): return np.sign(x) * np.sqrt(np.abs(x))

def doc_join_func(doc):
    total = []
    for sent in doc:
        total += " ".join(sent)
        total += ". "
    return "".join(total)


def mimportance(vector, dim, metric, avgTensor):
    x = metric.pairwise(np.vstack(
        [np.squeeze(avgTensor), np.squeeze(normalize(np.asarray(vector).reshape(-1, dim)))]))[0][1]
    if x != x:
        # print("nan")
        return -1
    if x > 1000.0:
        # print(x)
        return -1
    return x


def pairwiseproper(vector1, vector2, dim, metric, avgTensor):
    v1 = np.squeeze(normalize(np.asarray(vector1).reshape(-1, dim)))
    v2 = np.squeeze(normalize(np.asarray(vector2).reshape(-1, dim)))
    c = metric.pairwise(np.vstack([
        v1, v2
    ]))[0][1]
    if c != c: return -1
    return c


def pdist(vector1, vector2, dim, metric, avgTensor):
    a = mimportance(vector1, dim, metric, avgTensor)
    b = mimportance(vector2, dim, metric, avgTensor)
    c = metric.pairwise(np.vstack([
        np.squeeze(normalize(np.asarray(vector1).reshape(-1, dim))),
        np.squeeze(normalize(np.asarray(vector2).reshape(-1, dim)))]))[0][1]
    if a != a: return -1
    if b != b: return -1
    if c != c: return -1
    cosc = (a*a + b*b - c*c)/(2*a*b)
    return abs(1 - cosc)


def remaining(deepvec, closestvec, dim, metric, avgTensor, startvec = None):
    startvec = startvec or avgTensor
    dvec = mimportance(np.asarray([deepvec]), dim, metric, avgTensor)
    remaining = startvec
    for i in range(3):
        d1 = mimportance(np.asarray([closestvec]), dim, metric, avgTensor)
        d2 = mimportance((np.asarray(remaining).reshape(-1, 1)).T,  dim, metric, avgTensor)
        sig1 = sigmoid(d1 / (d1 + d2))
        sig2 = sigmoid(d2 / (d1 + d2))
        # s = sig(d1/(d1+d2))v1 + sig(d2/(d1+d2))v2
        remaining = (deepvec - sig1 * closestvec) / (1.0 - sig1)
    return remaining


def bnormalize(vec):
    return normalize(vec.reshape((-1, 1))).squeeze()
