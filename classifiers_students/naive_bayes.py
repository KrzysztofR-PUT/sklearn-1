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
        self.classes = []
        self.classProbabilities = dict()  # { Y : P(Y) }
        self.featureProbs = dict() # { { (Xindex, X, Y) : G(X) } }
        self.featureAvgs = dict()  # { (Xindex, Y) : val }
        self.featureDeviations = dict()

    def fit(self, X, y):
        self.classes = np.unique(y)
        numOfClasses = np.size(y)
        classQuantities = dict()
        for class_ in np.nditer(self.classes):
            classQuantities[class_.item(0)] = np.count_nonzero(y == class_)  # wykorzystac w linijce ponizej
            self.classProbabilities[class_.item(0)] = np.count_nonzero(y == class_) / float(numOfClasses)

        featuresWithClass = dict() # { (Xindex, Y) : occurances }

        for yIndex, y in np.ndenumerate(y):
            for xIndex in range(0, len(X[0,:])):
                if (xIndex, y) in featuresWithClass:
                    featuresWithClass[(xIndex, y)] = np.append(featuresWithClass[(xIndex, y)], X[yIndex, xIndex])
                else:
                    featuresWithClass[(xIndex, y)] = np.array(X[yIndex, xIndex])

        for key, value in featuresWithClass.iteritems():
            self.featureAvgs[key] = np.mean(value)
            self.featureDeviations[key] = np.std(value)


    def predict_proba(self, X):
        result = []
        for x in X:
            classScore = dict()
            for y in np.unique(self.classes):
                classScore[y] = self.classProbabilities[y]
                for index, xi in np.ndenumerate(x):
                    # print(self.featureDeviations[(index[0], y)])
                    classScore[y] *= 1 / (self.featureDeviations[(index[0], y)] * math.sqrt(2 * math.pi)) * math.exp(
                        ((-1) * math.pow(xi - self.featureAvgs[(index[0], y)], 2)) / (
                        2 * pow(self.featureDeviations[(index[0], y)], 2)))
            denominator = sum(classScore.values())
            res = np.array([])
            for score in classScore.values():
                res = np.append(res, [score/denominator])
            result.append(res)

        return np.array(result)

    def predict(self, X):
        result = []
        for x in X:
            classScore = dict()
            for y in np.unique(self.classes):
                classScore[y] = self.classProbabilities[y]
                for index, xi in np.ndenumerate(x):
                    # print(self.featureDeviations[(index[0], y)])
                    classScore[y] *= 1/(self.featureDeviations[(index[0], y)]*math.sqrt(2*math.pi))*math.exp(((-1)*math.pow(xi - self.featureAvgs[(index[0], y)],2))/(2*pow(self.featureDeviations[(index[0], y)],2)))
            result.append(max(classScore, key=classScore.get))
        return np.array(result)


class NaiveBayesNumNom(BaseEstimator):
    def __init__(self, is_cat=None, m=0.0):
        raise NotImplementedError

    def fit(self, X, yy):
        raise NotImplementedError

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError