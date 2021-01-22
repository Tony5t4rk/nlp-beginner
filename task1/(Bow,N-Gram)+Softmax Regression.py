import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split


class BagOfWord:
    def __init__(self):
        self.vocabulary_ = {}

    def fit(self, raw_documents):
        raw_documents = list(map(lambda x: str(x).strip().split(' '), raw_documents))
        word_set = set()
        for sentence in raw_documents:
            for word in sentence:
                word_set.add(word)
        elements = np.sort(list(word_set))
        labels = np.arange(len(elements)).astype(int)
        self.vocabulary_ = dict(zip(elements, labels))

    def transform(self, raw_documents):
        X = np.zeros((len(raw_documents), len(self.vocabulary_)))
        for i, sentence in enumerate(raw_documents):
            sentence = sentence.strip().split(' ')
            for word in sentence:
                if word in self.vocabulary_:
                    X[i][self.vocabulary_[word]] += 1
        return X

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)


class NGram:
    def __init__(self, ngram):
        self.ngram = ngram
        self.vocabulary_ = {}

    def fit(self, raw_documents):
        feature_set = set()
        for gram in self.ngram:
            for sentence in raw_documents:
                sentence = sentence.strip().split(' ')
                for i in range(len(sentence) - gram + 1):
                    feature = '_'.join(sentence[i:i + gram])
                    feature_set.add(feature)
        features = np.sort(list(feature_set))
        labels = np.arange(len(features)).astype(int)
        self.vocabulary_ = dict(zip(features, labels))

    def transform(self, raw_documents):
        X = np.zeros((len(raw_documents), len(self.vocabulary_)))
        for idx, sentence in enumerate(raw_documents):
            sentence = sentence.strip().split(' ')
            for gram in self.ngram:
                for i in range(len(sentence) - gram + 1):
                    feature = '_'.join(sentence[i:i + gram])
                    if feature in self.vocabulary_:
                        X[idx][self.vocabulary_[feature]] += 1
        return X

    def fit_transform(self, raw_documents):
        self.fit(raw_documents)
        return self.transform(raw_documents)


def softmax(x):
    x -= np.max(x, axis=1, keepdims=True)
    x = np.exp(x)
    x /= np.sum(x, axis=1, keepdims=True)
    return x


class SoftmaxRegression:
    def __init__(self, n_classes=2, learning_rate=0.1, max_iter=500):
        self.n_samples = None
        self.n_features = None
        self.n_classes = n_classes
        self.weights = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit(self, X, y):
        self.n_samples, self.n_features = X.shape
        self.weights = np.random.randn(self.n_classes, self.n_features)
        y = np.eye(self.n_classes)[y]  # ont_hot encode label
        loss_record = []
        for iter in tqdm(range(self.max_iter), desc='SoftmaxRegression.fit'):
            loss = 0
            probs = softmax(X.dot(self.weights.T))
            for i in range(self.n_samples):
                loss -= y[i].T.dot(np.log(probs[i] + 1e-6))
            loss /= self.n_samples
            loss_record.append(loss)
            grad = np.zeros_like(self.weights)
            for i in range(self.n_samples):
                grad += X[i].reshape(self.n_features, 1).dot((y[i] - probs[i]).reshape(self.n_classes, 1).T).T
            grad /= self.n_samples
            self.weights += self.learning_rate * grad
        return loss_record

    def prefict(self, X):
        probs = softmax(X.dot(self.weights.T))
        return probs.argmax(axis=1)


def main():
    data_df = pd.read_csv('./data/train.tsv', header=0, delimiter='\t')
    data_x, data_y = data_df['Phrase'].values, data_df['Sentiment'].values
    data_x, data_y = data_x[:10000], data_y[:10000]
    vectorizer = BagOfWord()
    # vectorizer = NGram(ngram=(1, 2, 3))
    data_x = vectorizer.fit_transform(data_x)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=66, stratify=data_y)
    model = SoftmaxRegression(n_classes=5, learning_rate=0.5, max_iter=1001)
    loss_record = model.fit(train_x, train_y)
    for loss in loss_record[::50]:
        print('loss: {}'.format(loss))
    preds = model.prefict(test_x)
    accuracy = np.mean(preds == test_y)
    print('accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
