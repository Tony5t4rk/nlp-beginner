import os
import spacy
import numpy as np
import pandas as pd
import torch
import torchtext
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

# device, GPU or CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
EPOCH_SIZE = 3
BATCH_SIZE = 128
CLASSES_SIZE = 5
LEARNING_RATE = 0.01
HIDDEN_SIZE = 128


def prepare_data(dataset_path, sentence_column_name, label_column_name):
    # load data
    data_df = pd.read_csv(dataset_path + os.sep + 'train.tsv', header=0, delimiter='\t')

    # extract data
    data_x = data_df[sentence_column_name].values
    data_y = data_df[label_column_name].values

    # splitting the training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2, random_state=66, stratify=data_y)

    # save as .csv
    train_df = pd.DataFrame({
        'sentence': train_x,
        'label': train_y
    })
    val_df = pd.DataFrame({
        'sentence:': val_x,
        'label': val_y
    })
    train_file = dataset_path + os.sep + 'train.csv'
    val_file = dataset_path + os.sep + 'validation.csv'
    train_df.to_csv(train_file, index=False)
    val_df.to_csv(val_file, index=False)
    # return train & validation .csv file
    return train_file, val_file


def load_data(dataset_path, sentence_column_name='Phrase', label_column_name='Sentiment', batch_size=32):
    # prepare data
    train_file, val_file = prepare_data(dataset_path, sentence_column_name, label_column_name)

    # load English tokenizer, tagger, parser, NER and word vectors
    spacy_en = spacy.load('en_core_web_sm')

    # define tokenizer
    def tokenizer(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    # build dataset
    text_field = torchtext.data.Field(sequential=True, tokenize=tokenizer, lower=True)
    label_field = torchtext.data.Field(sequential=False, use_vocab=False)
    train, val = torchtext.data.TabularDataset.splits(path='', train=train_file, validation=val_file,
                                                      format='csv', skip_header=True,
                                                      fields=[('sentence', text_field), ('label', label_field)])
    text_field.build_vocab(train, vectors='glove.6B.50d')
    text_field.vocab.vectors.unk_init = nn.init.xavier_uniform

    # create batch iterator
    train_iter = torchtext.data.BucketIterator(train, batch_size=batch_size, sort_key=lambda x: len(x.review),
                                               device='cuda' if torch.cuda.is_available() else 'cpu')
    val_iter = torchtext.data.BucketIterator(val, batch_size=batch_size, sort_key=lambda x: len(x.review),
                                             device='cuda' if torch.cuda.is_available() else 'cpu')
    # return train & validation iterator、word vectors、smaple size
    return train_iter, val_iter, text_field.vocab.vectors, len(train)


# Text RNN Model
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_classes, weights=None):
        super(TextRNN, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        if weights is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=weights)
        self.rnn = nn.RNN(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape
        embedding_output = self.embedding(x)
        h0 = torch.randn(1, batch_size, self.hidden_size).to(DEVICE)
        _, hn = self.rnn(embedding_output, h0)
        logits = self.output(hn).squeeze(0)
        return logits


# Text LSTM model
class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, n_classes, weights=None):
        super(TextLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.n_classes = n_classes

        if weights is None:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)
        else:
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, _weight=weights)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        batch_size, seq_len = x.shape
        embedding_output = self.embedding(x)
        h0 = torch.randn(1, batch_size, self.hidden_size).to(DEVICE)
        c0 = torch.randn(1, batch_size, self.hidden_size).to(DEVICE)
        output, (hn, _) = self.lstm(embedding_output, (h0, c0))
        logits = self.output(hn).squeeze(0)
        return logits


def main():
    # show device(GPU or CPU)
    print('DEVICE: {}'.format(DEVICE))

    # get train & validation iterator、word vectors、sample size
    train_iter, val_iter, word_vectors, sample_size = load_data(dataset_path='./data', batch_size=BATCH_SIZE)

    # define train model
    # model = TextRNN(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=HIDDEN_SIZE, n_classes=CLASSES_SIZE,
    #                 weights=word_vectors).to(DEVICE)
    model = TextLSTM(vocab_size=len(word_vectors), embedding_dim=50, hidden_size=HIDDEN_SIZE, n_classes=CLASSES_SIZE,
                     weights=word_vectors).to(DEVICE)
    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train
    for epoch in range(1, EPOCH_SIZE):
        model.train()
        for batch_idx, batch in enumerate(train_iter):
            x, y = batch.sentence.t().to(DEVICE), batch.label.to(DEVICE)
            logits = model(x)
            optimizer.zero_grad()
            loss = F.cross_entropy(logits, y)
            loss.backward()
            optimizer.step()
            # if batch_idx % 10 == 0:
            #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #         epoch, batch_idx * len(x), train_dataset_size,
            #                100. * batch_idx * len(x) / train_dataset_size, loss.item()))

    # calculate validation accuracy
    val_accuracys = []
    for batch_idx, batch in enumerate(val_iter):
        x, y = batch.sentence.t().to(DEVICE), batch.label.to(DEVICE)
        logits = model(x)
        _, preds = torch.max(logits, -1)
        accuracy = torch.mean((torch.tensor(preds == y, dtype=torch.float)))
        val_accuracys.append(accuracy.item())
    accuracy = np.array(val_accuracys).mean()
    print('Accuracy: {}'.format(accuracy))


if __name__ == '__main__':
    main()
