# Importing the Libraries
import pandas as pd
import numpy as np
import re, string
import swifter
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.multiclass import OneVsRestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.sparse import hstack
import joblib
import warnings
warnings.filterwarnings("ignore")

# Importing the training dataset
train_df = pd.read_csv("../input/toxic-data/train.csv")

# Importing the test data
test_data = pd.read_csv("../input/toxic-data/test.csv")
test_labels = pd.read_csv("../input/toxic-data/test_labels.csv")

# Merging the two datasets above for complete test data
test_df = pd.merge(test_data, test_labels, on="id")
test_df.head()

# Filtering out the samples having actual target labels
new_test_df = test_df[(test_df['toxic']!=-1) & (test_df['severe_toxic']!=-1) & (test_df['obscene']!=-1) & 
             (test_df['threat']!=-1) & (test_df['insult']!=-1) & (test_df['identity_hate']!=-1)]
new_test_df.reset_index(drop=True, inplace=True)
new_test_df.head()

# Creating a function to clean the training dataset
def clean_text(text):
    """This function will take text as input and return a cleaned text 
        by removing html char, punctuations, non-letters, newline and converting it 
        to lower case.
    """
    # Converting to lower case letters
    text = text.lower()
    # Removing the contraction of few words
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "can not ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
    # Replacing the HTMl characters with " "
    text = re.sub("<.*?>", " ", text)
    # Removing the punctuations
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    # Removing non-letters
    text = re.sub("[^a-zA-Z]", " ", text)
    # Replacing newline with space
    text = re.sub("\n", " ", text)
    # Split on space and rejoin to remove extra spaces
    text = " ".join(text.split())
    
    return text

def word_lemmatizer(text):
    """This function will help lemmatize words in a text.
    """
    
    lemmatizer = WordNetLemmatizer()
    # Tokenize the sentences to words
    text = word_tokenize(text)
    # Removing the stop words
    text = [lemmatizer.lemmatize(word) for word in text]
    # Joining the cleaned list
    text = " ".join(text)
    
    return text

# Cleaning and preprocessing the train data
train_df["comment_text"] = train_df["comment_text"].swifter.apply(clean_text)
train_df["comment_text"] = train_df["comment_text"].swifter.apply(word_lemmatizer)

# Cleaning and preprocessing the test data
new_test_df["comment_text"] = new_test_df["comment_text"].swifter.apply(clean_text)
new_test_df["comment_text"] = new_test_df["comment_text"].swifter.apply(word_lemmatizer)

# Performing the train-val split to create training and validation datasets
train, validation = train_test_split(train_df, test_size=0.2, random_state=42)
# print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)
print(train.shape, validation.shape)

# Seperating our input and target variable columns
X_train = train.comment_text
X_val = validation.comment_text
X_test = new_test_df.comment_text

# Storing our target labels list in a variable
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Creating a unigram TFIDF vectorizer and transforming all our input features
word_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 1), sublinear_tf=True, strip_accents="unicode", 
                             analyzer="word",token_pattern=r"\w{1,}", stop_words=stop_words)

word_tfidf.fit(train_df.comment_text)

train_word_tfidf = word_tfidf.transform(X_train)
val_word_tfidf = word_tfidf.transform(X_val)
test_word_tfidf = word_tfidf.transform(X_test)

# Creating a char n-gram (2, 6) TFIDF vectorizer and transforming all our input features
char_tfidf = TfidfVectorizer(max_features=30000, ngram_range=(2, 6), sublinear_tf=True, strip_accents="unicode", 
                             analyzer="char", stop_words=stop_words)

char_tfidf.fit(train_df.comment_text)

train_char_tfidf = char_tfidf.transform(X_train)
val_char_tfidf = char_tfidf.transform(X_val)
test_char_tfidf = char_tfidf.transform(X_test)

# Concatenating both unigram and n-gram features for our training input
train_features = hstack([train_word_tfidf, train_char_tfidf])
val_features = hstack([val_word_tfidf, val_char_tfidf])
test_features = hstack([test_word_tfidf, test_char_tfidf])

# Saving the tfidf vectors for future use
joblib.dump(word_tfidf, "word_tfidf_vectorizer.pkl")
joblib.dump(char_tfidf, "char_tfidf_vectorizer.pkl")

# Creating a logistic regression Model and treating each target as a binary classification problem
lr_model = OneVsRestClassifier(LogisticRegression(solver="saga"))
val_results = {"Accuracy": {}, "F1 Score": {}}
test_results = {"Accuracy": {}, "F1 Score": {}}

for label in labels:
    print(f"... Processing {label}")
    # train the model using X & y
    lr_model.fit(train_features, train[label])
    # Predicting the validation data labels
    val_prediction = lr_model.predict(val_features)
    # Predicting the test data labels
    test_prediction = lr_model.predict(test_features)
    # Saving the model based on target label
    joblib.dump(lr_model, f"logistic_regression_{label}.pkl")
    # Checking and model's accuracy and f1-score
    val_results["Accuracy"][f"{label}"] = accuracy_score(validation[label], val_prediction)
    val_results["F1 Score"][f"{label}"] = f1_score(validation[label], val_prediction, average = "weighted")
    test_results["Accuracy"][f"{label}"] = accuracy_score(new_test_df[label], test_prediction)
    test_results["F1 Score"][f"{label}"] = f1_score(new_test_df[label], test_prediction, average = "weighted")
    
