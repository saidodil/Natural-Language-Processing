# Natural-Language-Processing
UWT TCSS456 Coursework - NLP Toxic Text Classification Model
Model was developed as a TCSS456 (Natural Language Processing) course project by UW Tacoma students - Phuoc Le, Gil Rabara, Dilnoza Saidova, Vivian Tran.
June 4, 2023

The classification model implements CNN network architecture for deep learning to learn directly from data provided in csv files based on toxicity level of the text.

1. Introduction
  The problem addressed in the toxic email classification model is accurate identifying and classifying toxic emails based on their content. This is important because toxic emails can have negative consequences (i.e., harassment, bullying, or hate speech). Hence, it is crucial to detect and filter them effectively. However, this task is challenging due to the complexity and diversity of toxic language, including the use of slang, sarcasm, and cultural references. In addition to that, the lack of explicit indicators and the need to capture subtle contextual cues make it difficult to achieve high accuracy in classification.
  Previous solutions have made attempts to address the problem by using various machine learning algorithms, such as traditional classifiers and recurrent neural networks. However, these approaches often struggle to capture the nuances of toxic language and may have limitations in handling different languages or new emerging forms of toxicity. By taking natural language processing a little further, this project aims to establish tools to enable a more professional environment when communicating through emails.
  Immediate roadblocks include creating a standard that would be acceptable in all scenarios and selecting a specific model that would fit these broad requirements.  It is also difficult finding a large dataset with enough amount of negative values for Cleanse to train with. The key components of the proposed approach are based on a convolutional neural network (CNN) architecture. CNNs excel at capturing local patterns and features in text data, making them suitable for toxic language detection. The model utilizes word embeddings to represent textual data, followed by multiple convolutional layers and pooling operations to extract relevant features. Finally, a classification layer is used to predict whether an email is toxic or not. The approach aims to improve upon previous solutions by leveraging the strengths of CNNs in text classification tasks and potentially enhancing performance in accurately identifying toxic emails.
  
2. Problem Statement
  For the next few years, there doesn't seem to be any decline in remote work. People become more reliant on the power of technology more every day. Humanity is entering an age where people spend more time communicating through screens than in-person. What this means in the work environment is that it's a lot easier to send an email without a second thought about online etiquette - netiquette. Today's models rely on the users to report these messages and are only limited to a simple word checker to search for any profanity. The goal of this project of the model is to create a model that is accessible to anyone to improve their experience when sending and receiving messages. The objective is to have the model be able to not only detect potentially toxic language in a message, but also to be able to provide an alternative solution that can be received in a better light. 
  
3. Datasets
  The Jigsaw Toxic Comment Classification Challenge dataset available on Kaggle is a collection of comments from Wikipedia’s talk page edits. The comments are labeled based on their degree of toxicity, with a focus on six categories: toxic, severe toxic, obscene, threat, insult, and identity hate. The dataset contains two CSV files: "train.csv" and "test.csv". The original "train.csv" file contains over 159,000 comments, along with their corresponding labels. The original "test.csv" file contains over 63,000 comments, but without their corresponding labels, as they are reserved for evaluation by the competition organizers.
  One issue with the dataset is that the labels were generated using a crowdsourcing platform, which means that they may not always be accurate. Additionally, the dataset contains comments with offensive and discriminatory language, which can be difficult to work with for some people. However, the dataset provides a unique opportunity to develop models that can detect and classify toxic language, which can be valuable for online content moderation and related applications. Overall, the dataset is a good resource for training models to identify toxic language and provides a starting point for researchers to develop more advanced models for text classification.
  
4.1. Model Description
  CNN, or Convolutional Neural Network, is a type of neural network commonly used in computer vision tasks, such as image classification, object detection, and segmentation. The key feature of a CNN is that it uses convolutional layers, which are designed to automatically learn and detect local features or patterns in an input image.
  In the toxic email classification project, a CNN is selected as the neural network architecture for a few reasons. Firstly, an email's text can be considered a 2D image with words and their sequence represented along one axis and the different emails represented along the other axis. By using a 2D convolutional layer, the model can learn patterns or features in the text, such as character and word-level combinations and sentence-level structures. Additionally, CNNs are good at learning relevant features automatically, which is useful in natural language processing tasks where feature engineering can be time-consuming and complicated. By using convolutional layers, the model can learn both low-level and high-level features from the text, improving the accuracy of the classification task.

4.2. Model Architecture
  The CNN architecture for toxic email classification consists of several convolutional layers followed by max-pooling layers to extract local features from the text data. The output of the convolutional layers is then flattened and fed into a fully connected layer with a sigmoid activation function to output a binary classification for each email. The model uses the binary cross-entropy loss funcation and the Adam optimizer with a default learning rate of 0.001. The input sequences are preprocessed by lowercasing, removing punctuation, stemming, and removing special characters before being passed through the model. The below architecture, used for the model development, is a commonly used and effective architecture for text classification tasks:
  a. Input layer takes an input of shape (300,), which is the maximum length of the padded sequences.
  b. Embedding layer maps each word index to a dense vector of size 300.
  c. 1D Convolutional Layer with kernel size 3 and 5 filters with a ‘tanh’ activation function and ‘same’ padding.
  d. Dropout layer with 0.5 rate.
  e. Bidirectional LSTM layer with 200 unit in each direction, dropout rate of 0.5, and recurrent dropout rate of 0.25.
  f. Concatenation of the GlobalAverage Pooling 1D and GlobalMaxPooling1D layers.
  g. Fully connected output layer with 6 units and sigmoid activation function.
  
5.1. Model Compilation & Training 
  The CNN model is compiled with an appropriate loss function (binary cross-entropy), an optimizer (Adam), and a metric (accuracy). The model is then be trained on the preprocessed and split dataset using the fit() function in Keras with the appropriate batch size and number of epochs. 
  1 # Build and train model
  2 inp = Input(shape=(300,))
  3 x = Embedding(5000, 300, trainable=True)(inp)
  4 x = Conv1D(kernel_size=3, filters=5, padding='same', activation='tanh', strides=1)(x)
  5 x = Dropout(0.5)(x)
  6 x = Bidirectional(LSTM(200, return_sequences=True, dropout=0.5, recurrent_dropout=0.25))(x)
  7 x = concatenate([GlobalAveragePooling1D()(x), GlobalMaxPooling1D()(x)])
  8 out = Dense(6, activation='sigmoid')(x)
  9 model = Model(inp, out)
  10 model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

5.2. Experiments
  The data is split up as follows: 16,000 (80%) for training and 4,000 (20%) testing. Following the Pareto  Principle, the project was designed and created to provide the most impact and influence with the minimum amount of work.
  In the earlier life of the project, it was given a 50/50 split which yielded a lower accuracy with the higher the filter size and 1 epoch.
    Filter size 5 - Accuracy 70.38%
    Filter size 10 - Accuracy 46.86%
    Filter size 15 - Accuracy 7.8%
  After giving the project at least 2 epochs and adjusting the split size, the model was able to reach much higher accuracy. The model performed better on teh second epoch regardless of the filter size with highest accuracy at filter size 15 and 2 epochs - 94.27% (Validation accuracy = 99.5%).
    Filter size 5, Epoch 1/2 - Accuracy 51.03%
    Filter size 5, Epoch 2/2 - Accuracy 95.03%
    Filter size 10, Epoch 1/2 - Accuracy 84.88% 
    Filter size 10, Epoch 2/2 - Accuracy 93.12%
    Filter size 15, Epoch 1/2 - Accuracy 55.88%
    Filter size 15, Epoch 2/2 - Accuracy 94.27%
    
6. Evaluation Results
  Training data instances - 16,000 (80%)
  Training data instances - 4,000 (20%)
  From the data tables directly below, there is a correlation between the batch size, loss, accuracy and epochs. Turning attention to batch number and loss, the pattern emerges the more batch numbers the result is less loss. With this pattern continuing the project will continue to get a higher accuracy reading.  This correlation can be seen even further with the graphs below. A surprising finding is that with one epoch from batch 21 to 25 there's a slight decline the accuracy of correct readings, then is gradually increases again.  Addressing the first epoch out of 2 the same issue arises again where batch 21 to 25 there's a slight decline then a gradual rise to correct readings.  What's the most interesting between each first epoch is that the 1/1 spikes at 3 batches  then declines until batch 9 , however, the overall accuracy is higher than the 1/2 epoch.  Looking at the averages from 1 to 9 batch sizes there's a huge disparity which is interesting to say the least.
  
![Graphs Summary](https://github.com/saidodil/Natural-Language-Processing/assets/73456940/0dea0ddd-4726-4535-ba0e-8c1dfe42d58e)
  Training and Validation Loss: Since both, the training and validation loss decrease steadily and converge to a low value, it indicates that the model is learning effectively and fitting the data well.
  Training and Validation Accuracy: Since both, the training and validation accuracy increase consistently and converge to a high value, it indicates that the model is learning effectively and performing well on both, the training and validation data.
  Overall, a good model performance is characterized by decreasing loss and increasing accuracy for training and validation data. The goal is to strike a balance between fitting the training data well and generalizing to new data. Since there is no significant gap between the training and validation performance, such as high training accuracy but low validation accuracy, it indicates there is no overfitting.

7. Conclusion and Future Work: 
  Stated in the above results, the model is affective for only detecting text based comments such as emails, chats, and other forms of messages. Toxic comments are not limited to just the realms of text, they can also appear in other forms such as pictures and audio. The model also could find some complications when dealing with higher level intricacies of the English language such as sarcasm and humor due to the abstraction and implications required to understand these forms of text. Another limitation to the model is different languages other than English. The outcome is not defined if a toxic comment would be in an entirely different language. 
  Looking towards the bright side of the project, the reason the team decided on this project was for the positive impact it would make specifically in the work place. The data set that we collected data from was pulled from a company that was notorious for toxic emails as such. The team believed that in a professional setting there is no room for degrading one another that would harbor the roots for a toxic environment. In the future this project could also contain the capabilities to  process 3D images given the growing amount of media consumed by individuals on a daily basis. Some social media platforms directly share 3D images such as Instagram and Snapchat. Although this would be quite a hefty integration it would expand the impact of the project further than possible before. 

  
