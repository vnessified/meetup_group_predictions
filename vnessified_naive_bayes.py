from collections import Counter, defaultdict
import numpy as np

class vnessifiedNB(object):

    def __init__(self, alpha=1):
        '''
        My own implementation of NaiveBayes
        INPUT:
        - alpha: float, laplace smoothing constant
        '''

        self.class_totals = None
        self.class_feature_totals = None
        self.class_counts = None
        self.alpha = alpha

    def _compute_likelihood(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        Compute the totals for each class and the totals for each feature
        and class.
        '''
        # contain the sum of all the features for each class - this is S_y
        self.class_totals = Counter()

        # contain the sum of each feature for each class
        # this is S_yj
        # dictionary of dictionaries (technically a defaultdict of Counters)
        self.class_feature_totals = defaultdict(Counter)

        class_labels = X.shape[0]
        features = X.shape[1]

        # loop through the rows to get the classes
        for class_y in range(class_labels):
            # loop through the columns for the features
            for feature_j in range(features):

                # for class = 0, sum the corresponding rows - p(class0)
                # for class = 1, sum the corresponding rows - p(class1)
                self.class_totals[y[class_y]] += X[class_y, feature_j]

                # for each row where class = 0, get sum of each individual feature
                # for each row where class = 1, get sumeof each individual feature
                # This is essentially like a conditional probability table
                self.class_feature_totals[y[class_y]][feature_j] += X[class_y, feature_j]

    def fit(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT: None
        '''

        # This section is given to you.

        # compute priors
        self.class_counts = Counter(y)

        # compute likelihoods
        self._compute_likelihood(X, y)

    def predict(self, X):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix

        OUTPUT:
        - predictions: numpy array
        '''

        predictions = np.zeros(X.shape[0])

        class_labels = X.shape[0]
        features = X.shape[1]

        # loop through the rows to get the classes
        for class_y in range(class_labels):
            # get class_label count
            class_count = Counter()

            # get the prior probability of classes
            for key, value in self.class_counts.items():
                prior = Counter()

                # divide classes by total classes
                prior[key] = self.class_counts[key] / class_labels

                # update class_count with log of prior
                class_count[key] += np.log(prior[key])

                # for each row in the feature matrix X and for each potential label
                for feature_j in range(features):

                    # this is just the log likelihoood calcuation:
                    # log(S_yj + alpha / S_y + alpha * p)
                    class_count[key] += X[class_y, feature_j] * np.log(
                        (self.class_feature_totals[key][feature_j] + self.alpha)/
                           (self.class_totals[key] + self.alpha * features))

            # index into class
            predictions[class_y] = class_count.most_common(1)[0][0]

        return predictions

    def score(self, X, y):
        '''
        INPUT:
        - X: 2d numpy array, feature matrix
        - y: numpy array, labels

        OUTPUT:
        - accuracy: float between 0 and 1

        Calculate the accuracy, the percent of documents predicted correctly.
        '''
        return sum(self.predict(X) == y) / float(len(y))
