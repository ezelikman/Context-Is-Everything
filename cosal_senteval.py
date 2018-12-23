from vectormath import *
from weights import *
from docprocessing import *
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import normalize
import numpy as np
from scipy.spatial.distance import cdist
from pymagnitude import *
from random import uniform, choice, sample
from collections import Counter
import inspect


# The class that does the heavy lifting
class SentVecProcessor:
    def __init__(self, wordVectors, horizontal=4.6, vertical=5.1, slope=1.87, cutoff=0,
                 alphaInput=False, normalizeData=True, isLowercaseInput=True,
                 weightfn=meandistanceweight, localAverage="localline"):
        self.avgTensor, self.metric = None, None
        self.accept_dist = np.arange(0.01, 1.00, 0.001)
        self.sentVectors = [[]]
        self.wordVectors = wordVectors
        self.dim = wordVectors.dim
        self.horizontal = horizontal
        self.vertical = vertical
        self.localAverage = localAverage
        self.cutoff = cutoff
        self.alphaInput = alphaInput
        self.normalizeData = normalizeData
        self.weightfn = weightfn
        self.slope = slope
        self.printDistsWeightsAndElems = False
        self.isLowercaseInput = isLowercaseInput
        self.covar = None
        self.save = True
        # Returns the current "meaningfulness" of a word in context

    def distance(self, vector):
        return mimportance(np.asarray(vector).reshape(1, -1), self.dim, self.metric, self.avgTensor)

    # Calculates the Mahalanobis-cosine distance between two vectors
    def pairwisedistance(self, vector1, vector2):
        return pdist(vector1, vector2, self.dim, self.metric, self.avgTensor)

    # Calculates the Mahalanobis distance between two vectors
    def sentdistance(self, vector1, vector2):
        return pairwiseproper(vector1, vector2, self.dim, self.metric, self.avgTensor)

    def load(self, covfile, avgfile=None):
        self.covar = np.load(covfile)
        self.avgTensor = np.load(avgfile) if avgfile else np.zeros_like(self.covar)

    # # Updates the context using the training examples
    def prepare(self, params, samples):
        n = -1
        counts = Counter()
        for words in samples[:n]:
            # if sum(counts.values()) % 100 is 0: print(sum(counts.values()))
            words = [w for w in words if any(char.isalpha() or char.isdigit() for char in w)]  # Strips non-alpha 'words'
            for word in words:
                newword = word
                # newword = "".join(char for char in word if char.isalpha() or char.isdigit()).lower()
                if newword is "": continue
                counts[newword] += 1

        vecs = wordVectors.query(counts.keys())
        summer = sum(vecs[i] * count for i, count in enumerate(counts.values()))
        if self.covar is None:
            self.avgTensor = summer / float(sum(counts.values()))
            self.covar = np.cov(np.asarray(vecs).T, fweights=counts.values())
            self.metric = DistanceMetric.get_metric('mahalanobis', V=self.covar)
        else:
            textcov = np.cov(np.asarray(vecs).T, fweights=counts.values())
            p = 0.2
            self.covar = sign_root(self.covar) * np.sqrt(p*np.abs(textcov) + (1 - p)*np.abs(self.covar))
            self.metric = DistanceMetric.get_metric('mahalanobis', V=self.covar)

        if self.save:
            np.save("cov100d", self.covar)
            np.save("avg100d", self.avgTensor)

    # Calculates the current element vector from subelements
    def get_vector(self, structure):
        if isinstance(structure, basestring):
            try:
                return self.wordVectors.query(structure.lower() if self.isLowercaseInput else structure).reshape(1, -1)
            except Exception as e:
                print("vector not found:", e)
                return self.avgTensor
        return self.convolve(structure)

    def convolve(self, elems):
        if self.alphaInput:
            elems = [elem for elem in elems if not isinstance(elem, basestring) or any(char.isalpha() for char in elem)]
        if not elems: return self.avgTensor
        if len(elems) is 1: return self.get_vector(elems[0])
        vecs = [self.get_vector(elem) for elem in elems]

        if self.localAverage is "local":
            sentVec = np.average(vecs, axis=0)
            ds = np.asarray([self.sentdistance(sentVec, vec) for vec in vecs])
            ws = self.weightfn(ds, vecs, self.distance, self.horizontal, self.vertical, self.cutoff)
        elif self.localAverage is "bow": ws = np.ones(len(vecs)) / 2.0
        elif self.localAverage is "localline":
            sentVec = normalize(np.average(vecs, axis=0))
            ds = np.asarray([self.pairwisedistance(sentVec, vec) for vec in vecs])
            ds = harmonicscale(ds)
            ws = self.slope * (ds - 0.5) + 0.5
        elif self.localAverage is "localcosine":
            sentVec = np.average(vecs, axis=0)
            ds = np.asarray([self.pairwisedistance(sentVec, vec) for vec in vecs])
            ws = self.weightfn(ds, vecs, self.distance, self.horizontal, self.vertical, self.cutoff)
        elif self.localAverage is "global":
            ds = np.asarray([self.distance(vec) for vec in vecs])
            ws = self.weightfn(ds, vecs, self.distance, self.horizontal, self.vertical, self.cutoff)

        if self.printDistsWeightsAndElems:
            print elems, ds, ws

        result = [w * vecs[i] for i, w in enumerate(ws)]
        retVal = np.sum(result, axis=0)
        if self.normalizeData: retVal = normalize(retVal)
        return retVal

    def batcher(self, params, batch):
        return np.vstack(map(self.get_vector, batch))  # Returns the batch


if __name__ == "__main__":
    devmode = False
    printSource = False
    VECTOR_PATH = "crawl-300d-2M.magnitude.magnitude"
    print(VECTOR_PATH)
    wordVectors = Magnitude(VECTOR_PATH, lazy_loading=-1)
    processor = SentVecProcessor(wordVectors)

    from senteval import engine

    TASK_PATH = "/Users/eric/Documents/veridicality/data/downstream/senteval_data"
    params = {'task_path': TASK_PATH, 'usepytorch': False, 'kfold': 10}
    # params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128, 'tenacity': 3, 'epoch_size': 2}
    # params['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64, 'tenacity': 10, 'epoch_size': 8, 'dropout': 0.5}

    se = engine.SE(params, processor.batcher, processor.prepare)
    transfer_tasks = [
        # 'IMDB',
        # 'SICKEntailment',
        # 'MRPC',
        # 'MR',
        # 'MPQA',
        # 'SST2',
        # 'CR',
        # 'SUBJ',
        'TREC',
    ]

    if printSource:
        print(inspect.getsource(SentVecProcessor))
    for task in transfer_tasks:
        results = se.eval([task])
        if devmode:
            print task, results[task]['devacc']
        else:
            print task, results[task]
