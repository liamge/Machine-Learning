__author__ = 'liamgeron'
import collections, nltk

"""
See more at NLTK Chapter 6
P (label|features) = P(features, label)/P(features)
P(features) is consistent for each label, so it suffices to calculate P(features, label)
P(features, label) = P(label) x P(features|label)
"""
class NaiveBayes:
    def __init__(self, arfffile):
        self.trainingFile = arfffile
        self.features = {}
        self.featureNameList = []
        self.featureCounts = collections.defaultdict(lambda: 1) # For smoothing
        self.featureVectors = []
        self.labelCounts = collections.defaultdict(lambda: 0)





def feature_extract(sentence, i):
    # Extract features of interest where sent is observed states and i is the index
    features = {}

    return features

featuresets = []
size = int(len(featuresets) * 0.1)
train_set, test_set = featuresets[size:], featuresets[:size]