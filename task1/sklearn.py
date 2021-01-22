import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def main():
    data_df = pd.read_csv('./data/train.tsv', header=0, delimiter='\t')
    data_x, data_y = data_df['Phrase'], data_df['Sentiment']
    data_x, data_y = data_x[:10000], data_y[:10000]
    vectorizer = CountVectorizer()
    data_x = vectorizer.fit_transform(data_x)
    train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.2, random_state=66, stratify=data_y)
    model = LogisticRegression(max_iter=1001)
    model.fit(train_x, train_y)
    preds = model.predict(test_x)
    accuracy = np.mean(preds == test_y)
    print('accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
