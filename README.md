# Automatic-Ticket-Classification-using-NLP
This is the Capstone Project I did as part of my Data Science Course.
# Problem Statement
Cluster the complaints received by a financial institution in to 5 categories. Create a model that can classify the tickets automatically which will help route the complaint to appropriate department and resolve it faster.

# Process and Techniques used 
Exploratory Data Analysis

Text Processing

Clustering

Word Cloud Visualisation

Test Train Split the Data

Vectorization - Count Vectorizer and TFID Transformer

Apply Standard Scaler and Dimensionality Reduction

Modelling And Evaluation

Conclusion

# Files 
Dataset in JSON file format acquired from Kaggle

3 different Jupyter notebooks 1) Automatic_Ticket_Classification_Capstone_Project.ipynb; 2)EDA2 - Capstone.ipynb; 3)Modelling_Capstone_3.ipynb

Detailed Project Documentation Report - "Capstone Project_Documentation_Report"

Project Presentation Slides - "Presentation - Automatic Ticket Classification - Using NLP"


# Libraries or Packages used in this project:
Below is the code used to import libraries (this will be helpful to understand where the packages are imported from)
Note: pip install is required to import some libraries, this is not shown in the notebook as they have been already installed on the local computer.

import numpy as np

import pandas as pd

import json

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

import nltk

import spacy

from spacy.tokenizer import Tokenizer

import en_core_web_md

from sklearn.model_selection import train_test_split

import re

from spacy import displacy

import string

from collections import Counter

from wordcloud import WordCloud

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

from sklearn.pipeline import Pipeline

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score

from sklearn.metrics import precision_score

from sklearn.metrics import recall_score

from sklearn.metrics import average_precision_score

from sklearn.metrics import roc_auc_score

from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve

from sklearn.metrics import aucfrom sklearn import metrics

from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import AdaBoostClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import RandomizedSearchCV

from sklearn.utils import shuffle

import warnings
warnings.filterwarnings('ignore')

