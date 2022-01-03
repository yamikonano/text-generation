import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output
import pandas as pd
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords # Import the stop word list
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn import model_selection, naive_bayes, svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, ShuffleSplit
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix,roc_curve
np.random.seed(500)

# Add the Data using pandas
print("load data from file...")
rawInputFile = "input.csv"
tmpInputFile = "inputBoW_tmp.csv"

# only keep 3000 lines fro Medium, High, Low vulnerabilities
vulRiskLevel2count = {
    "HIGH": 0,
    "MEDIUM": 0,
    "LOW": 0
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
            if vulRiskLevel2count[riskLevelKey] < 3000:
                fdst.write(line)
                vulRiskLevel2count[riskLevelKey] = vulRiskLevel2count[riskLevelKey] + 1
fsrc.close()
fdst.close()
print(vulRiskLevel2count)

train = pd.read_csv(tmpInputFile)
print("The shape of our data:",train.shape,"\n")
# print columns names
print("Our column names are:",train.columns.values)


def review_to_words( raw_review ):
    # Function to convert a raw review to a string of words
    # The input is a single string (a raw movie review), and
    # the output is a single string (a preprocessed movie review)
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review,"html.parser").get_text()
    #
    # 2. Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", review_text)
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))
    #
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    #
    # 6. Join the words back into one string separated by space,
    # and return the result.
    return( " ".join( meaningful_words ))

num = train['DESC'].size
clean_desc = []
for i in range (0, num):
    clean_desc.append(review_to_words(train['DESC'][i]))

Encoder = LabelEncoder()
Encoder.fit(train['label'])
Train_Y_Raw = Encoder.transform(train['label'])

print("Creating the bag of words...\n")
vectorizer = CountVectorizer(analyzer = "word",
                             tokenizer = None,
                             preprocessor = None,
                             stop_words = None,
                             max_features = 5000)

train_data_features = vectorizer.fit_transform(clean_desc)
train_data_features = train_data_features.toarray()
vocab = vectorizer.get_feature_names_out()
# print(vocab)

# Sum up the counts of each vocabulary word
dist = np.sum(train_data_features, axis=0)

# For each, print the vocabulary word and the number of times it
# appears in the training set
# for tag, count in zip(vocab, dist):
#     print (count, tag)
# print(train_data_features)
# print( "Training the random forest...")
# Initialize a Random Forest classifier with 100 trees
# forest = RandomForestClassifier(n_estimators = 100)
# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
# forest = forest.fit( train_data_features, Train_Y_Raw)

print("\nTrain and test model...")
# Step - 5: Now we can run different algorithms to classify out data check for accuracy
# Classifier - Algorithm - Naive Bayes
# fit the training dataset on the classifier
acc = []
pre = []
rec = []
f1c = []

Naive = naive_bayes.MultinomialNB()
SVM = svm.SVC(C=1, kernel='linear', probability=True)
rfc = RandomForestClassifier(max_depth=2, random_state=0)
dtc = DecisionTreeClassifier(random_state=0)
lgc = LogisticRegression(max_iter=5000, random_state=0)
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
names = [
    # 'Naive', 'SVM', 'Random Forest', 'Decision Tree',
    'Logistic Regression']
sampling_methods = [Naive, SVM, rfc, dtc, lgc]
colors = ['crimson',
          'orange',
          'gold',
          'mediumseagreen',
          'steelblue',
          'mediumpurple'
          ]
# for model in [Naive, SVM, rfc, dtc, lgc]:
for (name, method, colorname) in zip(names, sampling_methods, colors):
    for train_index, test_index in cv.split(train_data_features, Train_Y_Raw):
        X_train, X_test = train_data_features[train_index], train_data_features[test_index]
        y_train, y_test = Train_Y_Raw[train_index], Train_Y_Raw[test_index]
        method.fit(X_train, y_train)
        y_test_preds = method.predict(X_test)
        # y_test_predprob = method.predict_proba(X_test)
    # fpr, tpr, thresholds = roc_curve(y_test, y_test_preds, pos_label=2, sample_weight=None, drop_intermediate=True)

#     plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, metrics.auc(fpr, tpr)), color=colorname)
#     plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
#     plt.axis('square')
#     plt.xlim([0, 1])
#     plt.ylim([0, 1])
#     plt.xlabel('False Positive Rate', fontsize=5)
#     plt.ylabel('True Positive Rate', fontsize=5)
#     plt.title('ROC Curve', fontsize=5)
#     plt.legend(loc='lower right', fontsize=5)
# plt.show()
# for train_index, test_index in cv.split(train_data_features, Train_Y_Raw):
#     X_train, X_test = train_data_features[train_index], train_data_features[test_index]
#     y_train, y_test = Train_Y_Raw[train_index], Train_Y_Raw[test_index]
#
#     lgc.fit(X_train, y_train)
#     y_pred_class = lgc.predict(X_test)
#     acc.append(accuracy_score(y_test, y_pred_class))
#     pre.append(precision_score(y_test, y_pred_class, average='weighted'))
#     rec.append(recall_score(y_test, y_pred_class, average='weighted'))
#     f1c.append(f1_score(y_test, y_pred_class, average='weighted'))

# print(Naive, ' accuracy score: ', acc)
# acc = []
# print(Naive, ' precision score: ', pre)
# pre = []
# print(Naive, ' recall score: ', rec)
# rec = []
# print(Naive, ' f-1 score: ', f1c)
# f1c = []
a = pd.read_csv(r'2015_2017.csv',nrows =50)
num = a['DESC'].size
clean_test = []
for i in range (0, num):
    clean_test.append(review_to_words(a['DESC'][i]))

test_data_features = vectorizer.fit_transform(clean_desc)
test_data_features = test_data_features.toarray()
# print(clean_desc)
# print(clean_test)
# a['final'] = a['DESC'].map(text_preprocessing)
# Test_X = Tfidf_vect.transform(a['final'])
# Pred_Y_SVM = clf.predict(Test_X)

# print description
# Test_X_index2DescriptionTexts = {}
# for index, value in Test_X.items():
#     Test_X_index2DescriptionTexts[str(index)] = value
Test_Y_RawLabels = a['S']
Test_Y_PredLabels = Encoder.inverse_transform(y_test_preds)


index = 0
for test_X_text in test_data_features:
    print("\n>>>Test Sample:", clean_test[index])
    test_Y_rawLabel = Test_Y_RawLabels[index]
    Predicted_Label = Test_Y_PredLabels[index]
    print("Raw label:", test_Y_rawLabel, "\tPredicted label:", Predicted_Label)
    GeneratedSentence = "The BS of the vulnerability is " + str(Predicted_Label)
    print("Summary: " + GeneratedSentence)
    index = index + 1
    if index > 30:  # Only describe the top 50 samples in the test set
        break

# plt.show()
# cm = []
