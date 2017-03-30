import math
import arff
import timeit
from scipy.stats import chi2


def entropyWeights(labels, weights, classCount, classarff):
    totalWeight = sum(weights)
    if totalWeight == 0:
        return (0, None)
    mostFrequentClass = -1
    mostFrequentClassNumSamples = 0
    E = 0.0
    for i in range(classCount):
        c = sum([weights[_] for _ in range(len(labels)) if labels[_] == i])
        p = float(c) / totalWeight
        if p != 0.0:
            E -= p * math.log(p, 2)
        if c > mostFrequentClassNumSamples:
            mostFrequentClassNumSamples = c
            mostFrequentClass = i
    return (E, mostFrequentClass)


def entropyFromWeight(weightsFV):
    E = 0.0
    mostFrequentClass = -1
    mostFrequentClassNumSamples = 0
    weightTotal = sum(weightsFV)
    for i in range(len(weightsFV)):
        c = weightsFV[i]
        p = float(c) / weightTotal
        if p != 0.0:
            E -= p * math.log(p, 2)
        if c > mostFrequentClassNumSamples:
            mostFrequentClassNumSamples = c
            mostFrequentClass = i
    return (E, mostFrequentClass)


class Node():
    def __init__(self):
        self.depth = -1
        # if the node is not a leaf, store feature for spliting data
        self.feature = -1
        # if the node is a leaf, store class information
        self.clas = -1
        # For each different value of an attribute, add a child to the node
        self.children = []
        # For ? or None values, keep the most frequent feature value
        self.defaultValue = -1
        self.childIndex = -1


class DecisionTree:
    def __init__(self, train, test=None):
        self.minIG = 0.0001
        self.minWeight = 0.01
        self.attStart = 0
        self.dataarff = train['data']
        self.attributes = train['attributes'][:-1]
        self.classarff = train['attributes'][-1][1]
        self.attCount = len(self.attributes)
        self.classCount = len(self.classarff)
        self.sampleCount = len(self.dataarff)
        self.convert()
        for i in range(-1, 22, 1):
            self.maxDepth = -1
            self.nodeCount = 0
            if i == -1:
                self.prun = False
            else:
                self.prun = True
            self.q = 1 - (float(i) / 20)
            self.prunCount = 0
            print "For prun=%s ; q = %s" % (self.prun, self.q)
            self.start = timeit.default_timer()
            self.root = self.growTree(range(self.sampleCount),
                                      self.weights)
            self.stop = timeit.default_timer()
            print "Train Time : ", self.stop - self.start
            print "Train Perf:"
            self.start = timeit.default_timer()
            self.test(train)
            self.stop = timeit.default_timer()
            if test is not None:
                print "Test (train) Time : ", self.stop - self.start
                print "Test Perf:"
                self.start = timeit.default_timer()
                self.test(test)
                self.stop = timeit.default_timer()
                print "Test (test) Time : ", self.stop - self.start

            print ("For q = %s; "
                   "maxDepth = %s; "
                   "node Count = %s; "
                   "prun Count = %s"
                   % (self.q,
                      self.maxDepth,
                      self.nodeCount,
                      self.prunCount))

    def growTree(self, dataIndices, weights, usedAttIndices=[], depth=0):
        node = Node()
        node.depth = depth
        if len(dataIndices) == 0:
            print "NO DATA IN NODE"
            node.clas = None
            return node

        sampleCount = len(dataIndices)
        labels = [self.labels[i] for i in dataIndices]

        # Number of samples for each class
        (E, C) = entropyWeights(labels,
                                weights,
                                self.classCount,
                                self.classarff)
        if E == 0:
            node.clas = C
            return node
        weightTotal = sum(weights)

        # Maximum Information Gain
        mig = 0
        # Feature Index for maximum information gain
        migFeature = -1
        for i in range(self.attCount):
            if i in usedAttIndices:
                continue
            attribute = self.features[i]
            # Data Indices Separated by Feature Value
            dataSep = [[] for _ in range(attribute)]
            weightSep = [[] for _ in range(attribute)]
            # Data Indices with missing value for Feature
            dataMissing = []
            weightMissing = []
            # Number of samples for each Class for each Feature Value
            weightClassFV = [[0.0] * self.classCount for _ in range(attribute)]
            # Number of samples for each Class with missing values
            sampleMisCount = 0
            weightClassMis = [0.0] * self.classCount
            for l in range(len(dataIndices)):
                s = dataIndices[l]
                sample = self.data[s]
                if sample[i] is None:
                    for k in range(self.classCount):
                        if labels[l] == k:
                            weightClassMis[k] += weights[l]
                    dataMissing.append(s)
                    weightMissing.append(weights[l])
                    sampleMisCount += 1
                else:
                    for j in range(attribute):
                        if sample[i] == j:
                            for k in range(self.classCount):
                                if labels[l] == k:
                                    weightClassFV[j][k] += weights[l]
                                    break
                            dataSep[j].append(s)
                            weightSep[j].append(weights[l])
                            break
            InfoGain = 0
            valuesWithSamples = sum([1 for w in weightClassFV if sum(w) > 0])
            nonMissingSamples = (float(sampleCount - sampleMisCount) /
                                 sampleCount > float(7 - depth) / 10)
            if (nonMissingSamples and
                sampleCount - sampleMisCount > 0 and
                    valuesWithSamples > 1):
                weightMisTotal = sum(weightClassMis)
                InfoGain = E
                # Most Frequent Feature Value
                freqFV = -1
                weightFreqFV = 0.0
                weightFV = [0.0] * attribute
                for j in range(attribute):
                    # Sample count for feature value
                    weightFV[j] = float(sum(weightClassFV[j]))
                    if weightFV[j] > weightFreqFV:
                        weightFreqFV = weightFV[j]
                        freqFV = j
                    if weightFV[j] > 0:
                        EFS = weightFV[j] / weightTotal
                        (EF, CF) = entropyFromWeight(weightClassFV[j])
                        InfoGain -= EFS * EF

                # Add data with missing values to all separated data list
                # with weights proportional to the weight of feature value
                InfoGainN = InfoGain
                InfoGain = ((InfoGainN *
                             (weightTotal - weightMisTotal)) /
                            weightTotal)
                if len(dataSep) != attribute:
                    print "DS ERROR ", len(dataSep),
                    print "; len(fv): ", attribute
            # Keep information for the feature that
            # has highest Information Gain value
            if InfoGain > mig and InfoGain > self.minIG:
                mig = InfoGain
                migFeature = i
                migDataSep = dataSep
                migWeightSep = weightSep
                migWeightClassFV = weightClassFV
                maxFreqFV = freqFV
                migWeightFV = weightFV
                migWeightMisTotal = weightMisTotal
                migDataMissing = dataMissing
                migWeightMissing = weightMissing

        # If the highest IG is positive, there should be a feature index
        # Else, the tree will not grow, most frequent class will be set to node
        toGrow = False
        if migFeature >= 0:
            if self.prun:
                """
                CHI-SQUARE
                """
                devX = 0
                # Number of distinct feature values
                v = len(migWeightClassFV)
                # For each feature value
                for j in range(v):
                    # Sample Count For Feature Value
                    Dx = sum(migWeightClassFV[j])
                    if Dx > 0:
                        # for each class
                        for k in range(self.classCount):
                            # Samples in Class k
                            # that has the same value for Feature
                            px = migWeightClassFV[j][k]
                            px_ = ((float(self.sampleClass[
                                   k]) / float(self.sampleCount)) *
                                   float(Dx))
                            devX += (((px - px_) ** 2) / px_)
                p = 1 - chi2.cdf(devX, df=v - 1)
                if p <= self.q:
                    toGrow = True
                else:
                    self.prunCount += 1
            else:
                toGrow = True
            if toGrow:
                node.feature = migFeature
                for j in range(self.features[migFeature]):
                    dataSep = migDataSep[j]
                    weightSep = migWeightSep[j]
                    if len(dataSep) > 0:
                        # Add data with missingValue
                        weightMultiplier = (migWeightFV[j] /
                                            (weightTotal -
                                             migWeightMisTotal))
                        for l in range(len(migDataMissing)):
                            weig = migWeightMissing[l] * weightMultiplier
                            if weig >= self.minWeight:
                                dataSep.append(migDataMissing[l])
                                weightSep.append(weig)

                        child = self.growTree(dataSep,
                                              weightSep,
                                              usedAttIndices + [migFeature],
                                              depth + 1)
                    else:
                        child = Node()
                        self.nodeCount += 1
                        if depth > self.maxDepth:
                            self.maxDepth = depth
                        child.clas = C
                        child.depth = depth + 1
                        child.childIndex = j

                    node.defaultValue = maxFreqFV
                    node.children.append(child)
        if not toGrow:
            node.clas = C
        return node

    def convert(self):
        """ Convert values to attribute and class indices
        """
        self.data = []
        self.labels = []
        self.features = []
        self.sampleClass = [0] * self.classCount
        for i in range(self.attCount):
            attValues = self.attributes[i][1]
            self.features.append(len(attValues))

        for sample in self.dataarff:
            data = []
            for i in range(self.attCount):
                attribute = self.attributes[i]
                attValues = attribute[1]
                missingValue = True
                for j in range(len(attValues)):
                    if sample[i] == attValues[j]:
                        data.append(j)
                        missingValue = False
                        break
                if missingValue:
                    data.append(None)
            self.data.append(data)
            for j in range(self.classCount):
                if sample[-1] == self.classarff[j]:
                    self.labels.append(j)
                    self.sampleClass[j] += 1
                    break
        self.weights = [1.0] * self.sampleCount

    def test(self, arf):
        """ Arf is the test data set in arff format
            Classify each sample in arf
            self.M is the confusion matrix
            Count True Positive, False Positive, True Negative, False Negative
        """
        self.M = [[0] * self.classCount for _ in range(self.classCount)]
        numSamples = 0
        testData = arf['data']
        for sample in testData:
            predictIndex = self.classify(self.root, sample)
            real = sample[-1]
            realIndex = -1
            for i in range(self.classCount):
                if real == self.classarff[i]:
                    realIndex = i
            if predictIndex >= 0 and predictIndex < self.classCount:
                self.M[predictIndex][realIndex] += 1
            numSamples += 1
        acc = 0
        for i in range(len(self.M)):
            acc += self.M[i][i]
        print self.M
        print float(acc) / numSamples

    def classify(self, node, sample):
        if node.clas is not None and node.clas != -1:
            return node.clas
        value = sample[node.feature]
        attribute = self.attributes[node.feature]
        if value is None:
            if node.defaultValue >= 0:
                return self.classify(node.children[node.defaultValue], sample)
        for i in range(len(attribute[1])):
            if value == attribute[1][i]:
                return self.classify(node.children[i], sample)

    def tr(self, node):
        if node is None:
            return
        print '-' * node.depth,
        print ' D: ', node.depth,
        print ' ; CI: ', node.childIndex,
        print ' ; F: ', node.feature,
        print ' ; DV : ', node.defaultValue,
        print ' ; C: ', node.clas,
        print ' ; LC: ', len(node.children)
        for child in node.children:
            self.tr(child)


if __name__ == "__main__":
    train = arff.load(open('training_subsetD.arff'))
    test = arff.load(open('testingD.arff'))
    D = DecisionTree(train, test)
