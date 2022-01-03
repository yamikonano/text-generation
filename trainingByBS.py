# 整多幾個model compare
# balance the data
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report, confusion_matrix, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score
from ggplot import *
import matplotlib.pyplot as plt
import pickle
import math
from sklearn.model_selection import cross_val_score, ShuffleSplit, cross_val_predict
from sklearn.model_selection import KFold, StratifiedKFold
import warnings


warnings.filterwarnings("ignore")


def text_preprocessing(text):
    # Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
    text = text.lower()

    # Step - 1c : Tokenization : In this each entry in the corpus will be broken into set of words
    text_words_list = word_tokenize(text)

    # Step - 1d : Remove Stop words, Non-Numeric and perfom Word Stemming/Lemmenting.
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(text_words_list):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
        # The final processed set of words for each iteration will be stored in 'text_final'
    return str(Final_words)


np.random.seed(500)

# Add the Data using pandas
print("load data from file...")
rawInputFile = "intputBScore.csv"
tmpInputFile = "inputBS_tmp.csv"

# only keep 3000 lines fro Medium, High, Low vulnerabilities
vulRiskLevel2count = {
    "1.0": 0,
    "2.0": 0,
    "3.0": 0,
    "4.0": 0,
    "5.0": 0,
    "6.0": 0,
    "7.0": 0,
    "8.0": 0,
    "9.0": 0,
    "10.0": 0,
}

fsrc = open(rawInputFile, "r")
fdst = open(tmpInputFile, "w")
firstLine = True
for line in fsrc.readlines():
    if firstLine == True:
        fdst.write(line)
        firstLine = False
        continue
    tmpLine = line.strip("\n")
    for riskLevelKey in vulRiskLevel2count.keys():
        if tmpLine.endswith(riskLevelKey):
            if vulRiskLevel2count[riskLevelKey] < 100:
                fdst.write(line)
                vulRiskLevel2count[riskLevelKey] = vulRiskLevel2count[riskLevelKey] + 1
fsrc.close()
fdst.close()
print(vulRiskLevel2count)

d1 = pd.read_csv(tmpInputFile)
print("Pre-process the text...")
d1['DESC'].dropna(inplace=True)
# WordNetLemmatizer requires Pos tags to understand if the word is noun or verb or adjective etc. By default it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
d1['final'] = d1['DESC'].map(text_preprocessing)

# Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(d1['final'], d1['label'],
#                                                                     test_size=0.2, random_state=0)

Encoder = LabelEncoder()
# Encoder.fit(Train_Y)
Encoder.fit(d1['label'])
# Train_Y_Raw = Train_Y
Train_Y_Raw = Encoder.transform(d1['label'])
# Train_Y = Encoder.transform(Train_Y)
# Test_Y = Encoder.transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(d1['final'])
#
# Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Train_X_Tfidf = Tfidf_vect.transform(d1['final'])
# Test_X_Tfidf = Tfidf_vect.transform(Test_X)

print("\nTrain and test SVM...")
# SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto', probability=True)
# SVM.fit(Train_X_Tfidf, Train_Y)
# clf = svm.SVC(C=1, kernel='linear').fit(Train_X_Tfidf, Train_Y)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
# for train_index, test_index in cv.split(Train_X_Tfidf):
#     print("TRAIN:", train_index, "TEST:", test_index)
clf = svm.SVC(C=1, kernel='linear', probability=True)
for score in ['accuracy','precision_weighted', 'recall_weighted', 'f1_weighted','roc_auc']:
    scores = cross_val_score(clf, Train_X_Tfidf, Train_Y_Raw, cv=cv,scoring=score)
    print(score,end='')
    print(':',end='')
    print(scores)
clf.fit(Train_X_Tfidf, Train_Y_Raw)
Pred_SVM = cross_val_predict(clf, Train_X_Tfidf, Train_Y_Raw, cv=5)
print(Pred_SVM.ndim)
# roc_auc_score(Train_Y_Raw, clf.predict_proba(Train_X_Tfidf), multi_class='ovr')
# fpr, tpr, threshold = roc_curve(Train_Y_Raw, clf.predict_proba(Train_X_Tfidf)[:,1])
# roc_auc = auc(fpr, tpr)
#
#
# df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
# ggplot(df, aes(x = 'fpr', y = 'tpr')) + geom_line() + geom_abline(linetype = 'dashed')
# print(Pred_SVM)
# scores = cross_val_score(clf, Train_X_Tfidf, Train_Y_Raw, cv=5)
# clf.score(Test_X_Tfidf, Test_Y)


# Test_X_Tfidf = Tfidf_vect.transform(test_index)
# # Evaluating
# Pred_Y_SVM = clf.predict(test_index)
# print("SVM Accuracy Score -> ", accuracy_score(Test_Y, Pred_Y_SVM) * 100)
# print("SVM Precision Score -> ", precision_score(Test_Y, Pred_Y_SVM, average='weighted') * 100)
# print("SVM Recall Score -> ", recall_score(Test_Y, Pred_Y_SVM, average='weighted') * 100)
# print("SVM F-1 Score -> ", f1_score(Test_Y, Pred_Y_SVM, average='weighted') * 100)

a = pd.read_csv(r'2015_2017.csv',nrows =50)
a['final'] = a['DESC'].map(text_preprocessing)
Test_X = Tfidf_vect.transform(a['final'])
Pred_Y_SVM = clf.predict(Test_X)

# print description
# Test_X_index2DescriptionTexts = {}
# for index, value in Test_X.items():
#     Test_X_index2DescriptionTexts[str(index)] = value
Test_Y_RawLabels = a['BS']
Test_Y_PredLabels = Encoder.inverse_transform(Pred_Y_SVM)


index = 0
for test_X_text in Test_X:
    print("\n>>>Test Sample:", a['final'][index])
    test_Y_rawLabel = Test_Y_RawLabels[index]
    Predicted_Label = Test_Y_PredLabels[index]
    print("Raw label:", test_Y_rawLabel, "\tPredicted label:", Predicted_Label)
    GeneratedSentence = "The BS of the vulnerability is " + str(Predicted_Label)
    print("Summary: " + GeneratedSentence)
    index = index + 1
    if index > 30:  # Only describe the top 50 samples in the test set
        break
exit()

# saving encoder to disk
filename = 'labelencoder_fitted.pkl'
pickle.dump(Encoder, open(filename, 'wb'))

# saving TFIDF Vectorizer to disk
filename = 'Tfidf_vect_fitted.pkl'
pickle.dump(Tfidf_vect, open(filename, 'wb'))

# saving the both models to disk
filename = 'svm_trained_model.sav'
pickle.dump(clf, open(filename, 'wb'))

# filename = 'nb_trained_model.sav'
# pickle.dump(Naive, open(filename, 'wb'))

print("Files saved to disk! Proceed to inference.py")