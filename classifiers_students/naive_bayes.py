import numpy as np
import math
from scipy.stats import norm
from sklearn.base import BaseEstimator

#NOTES:
#Zrobic dict'y i zliczac wystapienia
#Liczniki normaliziuja



class NaiveBayesNominal:


    def __init__(self):
        self.classProbabilities = dict()                # { Y : P(Y) }
        self.featureConditionalProbabilities = dict()   # { (Xindex, X, Y) : P(X|Y) }
        self.classes = []
        # self.classes_ = None
        # self.model = dict()
        # self.y_prior = []


    # X = 'dreszcze', 'katar', 'bol_glowy', 'goraczka'
    # y = 'grypa'

    def fit(self, X, y):
        self.classes = np.unique(y)
        numOfClasses = np.size(y)
        classQuantities = dict()
        for class_ in np.nditer(self.classes):
            classQuantities[class_.item(0)] = np.count_nonzero(y == class_) #wykorzystac w linijce ponizej
            self.classProbabilities[class_.item(0)] = np.count_nonzero(y == class_) / float(numOfClasses)

        for xIndex in range(0, len(X[0,:])):
            classesWithFeature = dict() # { (X,Y) : occurances }
            for yIndex in range(0, numOfClasses):
                classWithFeature = ( X[yIndex, xIndex], y[yIndex] )
                if classWithFeature not in classesWithFeature:
                    classesWithFeature[classWithFeature] = 1
                else:
                    classesWithFeature[classWithFeature] += 1

            for tup in classesWithFeature:
                self.featureConditionalProbabilities[(xIndex,) + tup] = classesWithFeature[tup] / float(classQuantities[tup[1]])

    def predict_proba(self, X):
        result = []
        for x in X:
            classScore = dict()
            for y in self.classes:
                classScore[y] = self.classProbabilities[y]
                for index, xi in np.ndenumerate(x):
                    classScore[y] *= self.featureConditionalProbabilities[(index[0], xi, y)]
            denominator = sum(classScore.values())
            result.append(max(classScore.values())/denominator)
        return np.array(result)


    def predict(self, X):
        result = []
        for x in X:
            classScore = dict()
            for y in self.classes:
                classScore[y] = self.classProbabilities[y]
                for index, xi in np.ndenumerate(x):
                    classScore[y] *= self.featureConditionalProbabilities[(index[0], xi, y)]
            result.append(max(classScore, key=classScore.get))
        return np.array(result)

class NaiveBayesGaussian:
    def __init__(self):
        raise NotImplementedError

    def fit(self, X, y):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError