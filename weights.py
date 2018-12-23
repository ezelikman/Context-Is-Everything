from vectormath import sigmoid
from scipy.stats import hmean
from scipy.stats.mstats import gmean
import numpy as np

# No sigmoid, calculates the weight linearly as a function of distance
# Scaled down for better recurrent performance, e.g. in documents of sentences of words
def lineardistanceweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    return ds/(max(ds)+min(ds))


# The sigmoid approach is based on a word's proportion of two words,
# but in a longer non-recursive string a proportion will cause values to all be
# at the left side of the sigmoid.
#
# Instead, it is important to find a point of relative comparison
# Ideally, without straying too far from the original relationship
# Thus, expressing the "typicality" of the set of words in a sentence is useful,
# and a center-seeking measure of the distances can serve as a relative proxy
#
# However, there are concerns to balance.
# Computational: The average should be fast to compute
# Bias: The distribution of distances is not uniform.
#   Should larger or smaller values be preferred?
#   If so, how much bias should there be?

### Peak - end Distance ###
# Surprisingly effective distance metric for its simplicity.
# Technically order-dependent, so tends to result in more comprehensible sentences
def peakendweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    peak = max(ds)
    end = ds[ds > cutoff][-1]
    ws = sigmoid(ds / (peak + end))
    return ws


### Normal Mean Distance ###
# Performs fairly well,
def meandistanceweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    newds = ds / (2 * np.mean(ds))
    ws = sigmoid(newds, horizontal, vertical)
    return ws


def logmeanweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    logds = np.log(ds)
    meands = np.log(ds[ds > cutoff].reshape(-1,1))
    newds = logds / (2 * meands)
    ws = sigmoid(newds, horizontal, vertical)
    return ws


def rmsscale(ds):
    return ds / (2 * np.sqrt(np.mean(np.square(ds.reshape(-1,1)))))


def rmsweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    newds = ds / (2 * np.sqrt(np.mean(np.square(ds.reshape(-1,1)))))
    ws = sigmoid(newds, horizontal, vertical)
    return ws


def harmonicscale(ds):
    return ds / (2 * hmean(ds))


def harmonicweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    newds = ds / (2 * hmean(ds[ds > cutoff].reshape(-1,1))) if cutoff else ds / (2 * hmean(ds))
    ws = sigmoid(newds, horizontal, vertical)
    return ws

def geometricweight(ds, vecs, distance, horizontal, vertical, cutoff=0):
    newds = ds / (2 * gmean(ds[ds > cutoff].reshape(-1,1))) if cutoff else ds / (2 * hmean(ds))
    ws = sigmoid(newds, horizontal, vertical)
    return ws