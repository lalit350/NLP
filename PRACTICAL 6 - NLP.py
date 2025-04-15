#!/usr/bin/env python
# coding: utf-8

# 1.	Using the names corpus in NLTK, build a gender classifier that predicts whether a name is male or female based on the last letter of the name. Evaluate its accuracy.
# 
# 2.	Enhance the gender classifier by including features such as the first letter and the length of the name. Evaluate if these features improve the classifier's accuracy.
# 
# 3.	Using the movie_reviews corpus in NLTK, build a document classifier to categorize movie reviews as positive or negative. Evaluate its performance.
# 
# 4.	Implement a custom feature extractor for the movie review classifier that considers bigrams (pairs of consecutive words) in addition to unigrams (single words). Evaluate its impact on classification accuracy.
# 
# 5.	Build a Naive Bayes classifier using the names corpus to predict gender based on both the first and last letters of a name. Evaluate the model's accuracy.

# 1.	Using the names corpus in NLTK, build a gender classifier that predicts whether a name is male or female based on the last letter of the name. Evaluate its accuracy.

# In[1]:


import nltk
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
import random

# Make sure to download required NLTK resources
nltk.download('names')

# Extract features: last letter of the name
def gender_features(name):
    return {'last_letter': name[-1]}

# Load male and female names
male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Create labeled dataset using the last letter as a feature
labeled_names = [(name, 'male') for name in male_names] + [(name, 'female') for name in female_names]

# Shuffle the dataset to avoid any bias in ordering
random.shuffle(labeled_names)

# Extract features and labels
featuresets = [(gender_features(name), gender) for (name, gender) in labeled_names]

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.25, random_state=42)

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(5)


# 2.	Enhance the gender classifier by including features such as the first letter and the length of the name. Evaluate if these features improve the classifier's accuracy.

# In[3]:


import nltk
from nltk.corpus import names
from nltk import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
import random

# Make sure to download required NLTK resources
nltk.download('names')

# Extract multiple features: first letter, last letter, and length of the name
def gender_features(name):
    return {
        'first_letter': name[0],
        'last_letter': name[-1],
        'length': len(name)
    }

# Load male and female names
male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Create labeled dataset using the first letter, last letter, and length of the name as features
labeled_names = [(name, 'male') for name in male_names] + [(name, 'female') for name in female_names]

# Shuffle the dataset to avoid any bias in ordering
random.shuffle(labeled_names)

# Extract features and labels
featuresets = [(gender_features(name), gender) for (name, gender) in labeled_names]

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.25, random_state=42)

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(5)


# 3.	Using the movie_reviews corpus in NLTK, build a document classifier to categorize movie reviews as positive or negative. Evaluate its performance.

# In[4]:


import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
import random

# Make sure to download required NLTK resources
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess the data: Tokenize the reviews and extract features
def word_features(words):
    stop_words = set(stopwords.words('english'))
    # We consider only non-stop words as features
    return {word: True for word in words if word.isalpha() and word not in stop_words}

# Load the movie_reviews corpus
positive_reviews = movie_reviews.categories('pos')
negative_reviews = movie_reviews.categories('neg')

# Create a list of labeled reviews
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))

# Shuffle the documents to ensure a random distribution
random.shuffle(documents)

# Extract features for all reviews
featuresets = [(word_features(words), category) for (words, category) in documents]

# Split the data into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(10)


# 4.	Implement a custom feature extractor for the movie review classifier that considers bigrams (pairs of consecutive words) in addition to unigrams (single words). Evaluate its impact on classification accuracy.

# In[5]:


import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.util import bigrams
import random

# Make sure to download required NLTK resources
nltk.download('movie_reviews')
nltk.download('stopwords')
nltk.download('punkt')

# Preprocess the data: Tokenize the reviews and extract both unigrams and bigrams as features
def extract_features(words):
    stop_words = set(stopwords.words('english'))

    # Generate unigrams and bigrams, filtering out non-alphabetic words and stopwords
    unigrams = {word: True for word in words if word.isalpha() and word not in stop_words}
    bigram_list = bigrams(words)  # Generate bigrams (pairs of consecutive words)

    # Combine unigrams and bigrams
    bigram_features = {f'bigram_{bigram[0]}_{bigram[1]}': True for bigram in bigram_list}

    # Combine both unigrams and bigrams into one feature set
    features = {**unigrams, **bigram_features}

    return features

# Load the movie_reviews corpus
documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.words(fileid), category))

# Shuffle the documents to avoid any bias in ordering
random.shuffle(documents)

# Extract features for all reviews
featuresets = [(extract_features(words), category) for (words, category) in documents]

# Split the data into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy with Unigrams + Bigrams: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(10)


# 5.	Build a Naive Bayes classifier using the names corpus to predict gender based on both the first and last letters of a name. Evaluate the model's accuracy.

# In[7]:


import nltk
from nltk.corpus import names
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
import random

# Make sure to download the necessary NLTK resources
nltk.download('names')

# Feature extractor: first and last letter of the name
def gender_features(name):
    return {
        'first_letter': name[0],  # First letter of the name
        'last_letter': name[-1]   # Last letter of the name
    }

# Load male and female names from the NLTK names corpus
male_names = names.words('male.txt')
female_names = names.words('female.txt')

# Create a labeled dataset using the first and last letters of each name
labeled_names = [(name, 'male') for name in male_names] + [(name, 'female') for name in female_names]

# Shuffle the dataset to avoid any bias in ordering
random.shuffle(labeled_names)

# Extract features for all names
featuresets = [(gender_features(name), gender) for (name, gender) in labeled_names]

# Split the dataset into training and testing sets
train_set, test_set = train_test_split(featuresets, test_size=0.25, random_state=42)

# Train a Naive Bayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate accuracy
accuracy = nltk.classify.accuracy(classifier, test_set)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Show the most informative features
classifier.show_most_informative_features(5)

