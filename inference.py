import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
import pickle
import csv
import ast

# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)
    ns = ["_","-"]
    sw = stopwords.words('english')
    sw.extend(ns)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in sw and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


# Loading Label encoder
labelencode = pickle.load(open('labelencoder_fitted.pkl', 'rb'))

# Loading TF-IDF Vectorizer
Tfidf_vect = pickle.load(open('Tfidf_vect_fitted.pkl', 'rb'))

# Loading models
SVM = pickle.load(open('svm_trained_model.sav', 'rb'))
# Naive = pickle.load(open('nb_trained_model.sav', 'rb'))


# Inference
csv_file = open('2015_2017.csv')

# sample_text = pd.read_csv(r"2012_2017data.csv")
# sample_text["content"] = sample_text["Content"].astype("string")
# # sample_text_processed_vectorized = Tfidf_vect.transform([sample_text_processed])
# sample_text['final']=sample_text["DESC"].map(text_preprocessing)
# content=[]
# for line in sample_text['final']:
#     content.append(line)
    # print(content)

# print(content)
#
# for i in content:
#     prediction_SVM = SVM.predict(i)
#     print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM))

# sample_text=open("textdata.txt")
# sample_text[1] = sample_text[0].map(text_preprocessing)
#
# sample_text_processed = text_preprocessing(sample_text)
# for line in sample_text_processed:
    # print(line)
    # sample_text_processed = text_preprocessing(line)
    # print(type(sample_text_processed))
    # prediction_SVM = SVM.predict(sample_text_processed)
    # print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM)[0])


#     print(("predict:", labelencode.inverse_transform(prediction_SVM)))
# prediction_SVM = SVM.predict(sample_text["content"])
# prediction_Naive = Naive.predict(sample_text_processed)

# sample_text = "The Windows Forms (aka WinForms) component in Microsoft .NET Framework 1.0 SP3, 1.1 SP1, 2.0 SP2, 3.0 SP2, 4, and 4.5 does not properly initialize memory arrays, which allows remote attackers to obtain sensitive information via (1) a crafted XAML browser application (XBAP) or (2) a crafted .NET Framework application that leverages a pointer to an unmanaged memory"  \
#               "location, aka System Drawing Information Disclosure Vulnerability."
# sample_text_processed = text_preprocessing(sample_text)
# sample_text_processed_vectorized = Tfidf_vect.transform([sample_text_processed])
# #
# prediction_SVM = SVM.predict(sample_text_processed_vectorized)
# print("Prediction from SVM Model:", labelencode.inverse_transform(prediction_SVM)[0])
# print("Prediction from NB Model:", labelencode.inverse_transform(prediction_Naive)[0])
index = 0
for test_X_text in Test_X:
    print("\n>>>Test Sample:", test_X_text)
    test_Y_rawLabel = Test_Y_RawLabels[index]
    Predicted_Label = Test_Y_PredLabels[index]
    print("Raw label:", test_Y_rawLabel, "\tPredicted label:", Predicted_Label)
    GeneratedSentence = "The BS of the vulnerability is " + str(Predicted_Label)
    print("Summary: " + GeneratedSentence)
    index = index + 1
    if index > 50:  # Only describe the top 50 samples in the test set
        break
exit()


