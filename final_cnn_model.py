# Phuoc Le, Gil Rabara, Dilnoza Saidova, Vivian Tran
# June 4, 2023
# Toxic email (text) classification model implementing CNN.

import re
import string

import matplotlib.pyplot as plt
import pandas as pd
from keras.layers import Dense, Input, LSTM, Bidirectional, Conv1D
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from nltk.stem import SnowballStemmer


# Load data
train_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/train1.csv').dropna()
train_data = train_data[train_data.comment_text.apply(lambda x: x != "")]
test_data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/test1.csv').dropna()
test_data = test_data[test_data.comment_text.apply(lambda x: x != "")]

y = train_data[['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']].values


# Preprocess data
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

# Prepare data for modeling
X_train = train_data["comment_text"].str.lower()
X_test = test_data["comment_text"].str.lower()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(list(X_train))
tokenized_train = tokenizer.texts_to_sequences(X_train)
X_training = pad_sequences(tokenized_train, maxlen=300)
tokenized_test = tokenizer.texts_to_sequences(X_test)
X_testing = pad_sequences(tokenized_test, maxlen=300)

# Build and train model
inp = Input(shape=(300,))
x = Embedding(5000, 300, trainable=True)(inp)
x = Conv1D(kernel_size=3, filters=15, padding='same', activation='tanh', strides=1)(x)
x = Dropout(0.5)(x)
x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(x)
x = concatenate([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
out = Dense(6, activation='sigmoid')(x)
model = Model(inp, out)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Assuming you have the training history recorded during model training
history = model.fit(X_training, y, batch_size=256, epochs=10, validation_split=0.4)

# Extracting the training and validation loss and accuracy from the history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plotting the training and validation loss
epochs = range(1, len(train_loss) + 1)

plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'go-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'go-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Adjusting the layout and displaying the plot
plt.tight_layout()
plt.show()
