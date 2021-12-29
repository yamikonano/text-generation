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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import pickle
# import nltk
# import ssl
#
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
#
# nltk.download()
# Set Random seed
np.random.seed(500)

# Add the Data using pandas

d1 = pd.read_csv(r"input.csv")

# Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

# Step - 1a : Remove blank rows if any.
d1['DESC'].dropna(inplace=True)

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


d1['final'] = d1['DESC'].map(text_preprocessing)
print("processed")
# Step - 2: Split the model into Train and Test Data set
# by sklearn library
# training set 70%, test set 30%
# x --> predictor, y --> target
Train_X, Test_X, Train_Y, Test_Y = model_selection.train_test_split(d1['final'], d1['label'],
                                                                    test_size=0.3)

# Step - 3: Label encode the target variable  - This is done to transform Categorical data of string type in the data set into numerical values
Encoder = LabelEncoder()
Encoder.fit(Train_Y)
Train_Y = Encoder.transform(Train_Y)
Test_Y = Encoder.transform(Test_Y)

# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(d1['final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Step - 5: Now we can run different algorithms to classify out data check for accuracy

# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
# Naive = naive_bayes.MultinomialNB()
# Naive.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
# predictions_NB = Naive.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
# print("Naive Bayes Accuracy Score -> ", accuracy_score(predictions_NB, Test_Y) * 100)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto',probability=True)
SVM.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
print("predicting")
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print('Precision score: %.3f' % precision_score(predictions_SVM, Test_Y,average='macro'))
print('recall score: %.3f' % recall_score(predictions_SVM, Test_Y,average='macro'))
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
print('F1 Score: %.3f' % f1_score(predictions_SVM, Test_Y,average='macro'))


# # predict the labels on validation dataset
# predictions_SVM = SVM.predict(Test_X_Tfidf)
#
# svm_prob = SVM.predict_proba(Train_X_Tfidf)
# # probability for positive outcome is kept
# svm_prob = svm_prob[:,1]
#
# svm_auc = roc_auc_score(Test_Y, svm_prob,multi_class='ovo')
# print("SVM: AUROC = %.3f" %(svm_prob))
# fpr, tpr, _ =roc_curve(Test_Y, svm_prob)
#
# # plot ROC curve
# plt.plot(fpr, tpr, marker='.', label = "SVM (AUROC %0.3f)" %(svm_auc))
# plt.title('ROC curve of SVM')
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.legend()
# plt.show()



# Saving Encdoer, TFIDF Vectorizer and the trained model for future infrerencing/prediction

# saving encoder to disk
filename = 'labelencoder_fitted.pkl'
pickle.dump(Encoder, open(filename, 'wb'))

# saving TFIDF Vectorizer to disk
filename = 'Tfidf_vect_fitted.pkl'
pickle.dump(Tfidf_vect, open(filename, 'wb'))

# saving the both models to disk
filename = 'svm_trained_model.sav'
pickle.dump(SVM, open(filename, 'wb'))

# filename = 'nb_trained_model.sav'
# pickle.dump(Naive, open(filename, 'wb'))

print("Files saved to disk! Proceed to inference.py")