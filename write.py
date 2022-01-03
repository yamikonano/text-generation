import pandas as pd
import csv
import math
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import defaultdict
from nltk.corpus import wordnet as wn
import nltk
# a=pd.read_csv(r"nvd_data_2012_2017_with_time_and_bid.csv")

# file = pd.read_csv(r"output.csv")
# writer = csv.writer(file)
a=pd.read_csv(r"nvd_data_2009_2015_with_time.csv")
# b=pd.read_csv(r"bs.csv")
# c=pd.read_csv(r"s.csv")
# print(round(a["BS"]))
# for i in a["BS"]:

with open('input.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(zip(a['DESC'],a['S']))