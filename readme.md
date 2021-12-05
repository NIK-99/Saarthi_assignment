This is an NLP Model which classify the input transcription into multiple classes namely action, object and location.

There were 6 labels for action, 14 labels for object and 4 labels for location as per the given data.
As the text in given dataset belongs to very gaint data space we could have used pre-trained models (transfer learning). But in this project word2vec, LSTM, etc are used to make a custom model.

These are the steps performed to achive the desired result

1. Organize the data in required format: The whole dataset is in text format, and we cannot feed text to model so we first need to convert the text into some numerical value
    a. The [action, object, location] are changed to numerical foamt by using One-Hot encoding 
    b. Transcrypt is cleaned of any stopwords, punctuation, etc and is made to be in lowecase
2. Traning the model : [action, object, location] these are taken as the target class which has to predicted and "Transcript" is taken as the X value
    The model cosist of embedding_layer, lstm layer, dense layer.
--------------------------------------------------------------------------------------------------------------------------------------------------------
# To run the model 
### Prerequisites:
- Python 3.8
- Tensorflow 2.5.0
- spacy
- Sklearn
- Pandas
- Numpy

run the following command in terminal  ```python main.py```

Colab link ```https://colab.research.google.com/drive/1crM3kPfhJcwHigvCVpTZ1DnCSSLluu_O?usp=sharing```







