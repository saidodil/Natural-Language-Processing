__name__ Phuoc Le, Gil Rabara, Dilnoza Saidova, Vivian Tran
__date__ May 28, 2023
__description__ Toxic email (text) classification model implementing CNN.

import re
import string
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D, Embedding, GlobalMaxPooling1D, \
    GlobalAveragePooling1D, concatenate, Dropout
from keras.models import Model
from nltk.stem import SnowballStemmer

# Load data.
train_data = pd.read_csv(
    '/content/drive/MyDrive/Toxic-Comment-Classification-Challenge-master/data/train1.csv').dropna()
train_data = train_data[train_data.comment_text.apply(lambda x: x != "")]
test_data = pd.read_csv('/content/drive/MyDrive/Toxic-Comment-Classification-Challenge-master/data/test1.csv').dropna()
test_data = test_data[test_data.comment_text.apply(lambda x: x != "")]


# Preprocess data.
def preprocess_data(raw_text):
    text = raw_text.translate(str.maketrans("", "", string.punctuation))
    text = text.lower().split()
    stemmed_words = [SnowballStemmer('english').stem(word) for word in text]
    text = " ".join(stemmed_words)
    text = (text.encode('ascii', 'ignore')).decode("utf-8")
    text = re.sub(r'[<>!#@$:.,%\?-]+', r'', text)
    text = re.sub(r'@\w+', r'', text)
    return text


train_data['comment_text'] = train_data['comment_text'].map(lambda x: preprocess_data(x))
test_data['comment_text'] = test_data['comment_text'].map(lambda x: preprocess_data(x))

# Prepare data for modeling.
X_train = train_data["comment_text"].str.lower()
X_test = test_data["comment_text"].str.lower()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(list(X_train))
tokenized_train = tokenizer.texts_to_sequences(X_train)
X_training = pad_sequences(tokenized_train, maxlen=300)
tokenized_test = tokenizer.texts_to_sequences(X_test)
X_testing = pad_sequences(tokenized_test, maxlen=300)

y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values

# Build and train model.
inp = Input(shape=(300,))
x = Embedding(5000, 300, trainable=True)(inp)
x = Conv1D(kernel_size=3, filters=10, padding='same', activation='tanh', strides=1)(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(x)
x = concatenate([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
out = Dense(6, activation='sigmoid')(x)
model = Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("Fitting...")
model.fit(X_training, y, batch_size=256, epochs=2, validation_split=0.4)
model.summary()
