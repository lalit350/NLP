#!/usr/bin/env python
# coding: utf-8

# 1. USE NLTK TO TOKENIZE THE SENTENCE
# 
# 2. USE NLTK TO FIND FREQUENCY DISTRIBUTION OF WORDS
# 
# 3. USE NLTK TO CREATE A BIGRAM COLLECTION FOR THE TEXT "SENSE AND SENSIBILITY" BY JANE AUSTEN AND LIST TOP 5 BIGRAMS
# 
# 4. USE NLTK ("SENSE AND SENSIBILITY" BY JANE AUSTEN), CALCULATE THE TOTAL NO.OF OF WORDS  AND NO.OF DISCTINCT WORDS
# 
# 5. COMPARE LEXICAL DIVERSITY  OF HUMOR AND ROMANCE FICTION IN NLTK'S TEXT 5 & 2. WHICH GENRE IS MORE LEXICALLY DIVERSE? 
# 
# 6. PRODUCE A DISPERSION PLOT OF THE 4 MAIN PROTAGONISTS IN "SENSE & SENSIBILITY": ELINOR, MARIANNE, EDWARD & WILLOUGHLY. WHAT      OBSERVATIONS CAN YOU MAKE ABOUT THEIR EXPERINCES IN THE TEXT?
# 
# 7. FIND THE COLLECTION IN NLTK TEXT5 (THE CHAT CORPUS). LIST TOP 5 COLLOCATIONS
# 
# 8. DEFINE THE TWO LISTS PHRASE1, PHRASE2, EACH CONTAINING A FEW WORDS. JOIN THEM TOGETHER TO FORM A SENTENCE

# 1. USE NLTK TO TOKENIZE THE SENTENCE

# In[1]:


import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('punkt_tab')


# In[3]:


sentence = "Natural language Processing with Python is fun!"
words = word_tokenize(sentence)
print(words)


# 2. USE NLTK TO FIND FREQUENCY DISTRIBUTION OF WORDS

# In[2]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('gutenberg')
nltk.download('punkt')


# In[4]:


# Load the text of Moby Dick from the Gutenberg corpus
moby_dick_text = nltk.corpus.gutenberg.raw('melville-moby_dick.txt')

# Tokenize the text into words
words = word_tokenize(moby_dick_text)

# Calculate the frequency distribution of words
fdist = FreqDist(words)

# Print the most common words
print(fdist.most_common(50))


# 3. USE NLTK TO CREATE A BIGRAM COLLECTION FOR THE TEXT "SENSE AND SENSIBILITY" BY JANE AUSTEN AND LIST TOP 5 BIGRAMS

# In[5]:


import nltk
from nltk.corpus import gutenberg
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures

# Download necessary NLTK resources if you haven't already
nltk.download('gutenberg')
nltk.download('punkt')


# In[6]:


# Load the text of "Sense and Sensibility"
text = gutenberg.raw('austen-sense.txt')

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Create a bigram collocation finder
finder = BigramCollocationFinder.from_words(tokens)

# Apply frequency filter to consider bigrams that occur at least 5 times
finder.apply_freq_filter(5)

# Get the top 5 bigrams using Pointwise Mutual Information (PMI) as the scoring measure
bigrams = finder.nbest(BigramAssocMeasures.pmi, 5)

# Print the top 5 bigrams
print("Top 5 Bigrams:")
for bigram in bigrams:
    print(bigram)


# 4. USE NLTK ("SENSE AND SENSIBILITY" BY JANE AUSTEN), CALCULATE THE TOTAL NO.OF OF WORDS  AND NO.OF DISCTINCT WORDS

# In[7]:


import nltk
from nltk.util import bigrams
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg

# Download necessary datasets
nltk.download('punkt')
nltk.download('gutenberg')


# In[8]:


# Load the text of "Sense and Sensibility"
text = gutenberg.raw('austen-sense.txt')

# Tokenize the text into words
tokens = nltk.word_tokenize(text)

# Calculate the total number of words
total_words = len(tokens)

# Calculate the number of distinct words
distinct_words = len(set(tokens))

# Print the results
print("Total number of words:", total_words)
print("Number of distinct words:", distinct_words)


# 5. COMPARE LEXICAL DIVERSITY  OF HUMOR AND ROMANCE FICTION IN NLTK'S TEXT 5 & 2. WHICH GENRE IS MORE LEXICALLY DIVERSE? 

# In[9]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg, webtext

# Download necessary datasets
nltk.download('punkt')
nltk.download('gutenberg')
nltk.download('webtext')


# In[10]:


# Load the text of "Sense and Sensibility" (text2) and humor fiction (text5)
text2 = gutenberg.raw('austen-sense.txt')
text5 = webtext.raw('singles.txt')

# Tokenize the texts into words
words_text2 = word_tokenize(text2)
words_text5 = word_tokenize(text5)

# Compute total and distinct words for both texts
total_words_text2 = len(words_text2)
distinct_words_text2 = len(set(words_text2))
lexical_diversity_text2 = distinct_words_text2 / total_words_text2

total_words_text5 = len(words_text5)
distinct_words_text5 = len(set(words_text5))
lexical_diversity_text5 = distinct_words_text5 / total_words_text5

# Print results
print("Total words in Sense and Sensibility:", total_words_text2)
print("Distinct words in Sense and Sensibility:", distinct_words_text2)
print("Lexical diversity in Sense and Sensibility:", lexical_diversity_text2)
print("Total words in Humor fiction:", total_words_text5)
print("Distinct words in Humor fiction:", distinct_words_text5)
print("Lexical diversity in Humor fiction:", lexical_diversity_text5)

# Compare lexical diversity
if lexical_diversity_text2 > lexical_diversity_text5:
    print("Romance fiction (Sense and Sensibility) is more lexically diverse.")
else:
    print("Humor fiction is more lexically diverse.")


# 6. PRODUCE A DISPERSION PLOT OF THE 4 MAIN PROTAGONISTS IN "SENSE & SENSIBILITY": ELINOR, MARIANNE, EDWARD & WILLOUGHLY. WHAT OBSERVATIONS CAN YOU MAKE ABOUT THEIR EXPERINCES IN THE TEXT?

# In[11]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import gutenberg
import matplotlib.pyplot as plt

# Download necessary datasets
nltk.download('punkt')
nltk.download('gutenberg')


# In[12]:


# Load the text of "Sense and Sensibility"
text = gutenberg.raw('austen-sense.txt')

# Tokenize the text into words
words = word_tokenize(text)

# Create an NLTK Text object
text_nltk = nltk.Text(words)

# Define the main protagonists
protagonists = ['Elinor', 'Marianne', 'Edward', 'Willoughby']

# Generate dispersion plot
plt.figure(figsize=(12, 6))
text_nltk.dispersion_plot(protagonists)
plt.show()


# 7. FIND THE COLLECTION IN NLTK TEXT5 (THE CHAT CORPUS). LIST TOP 5 COLLOCATIONS

# In[13]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import webtext

# Download necessary datasets
nltk.download('punkt')
nltk.download('webtext')
nltk.download('stopwords')


# In[14]:


# Load the text of the Chat Corpus (text5)
text5 = webtext.raw('singles.txt')

# Tokenize the text into words
words_text5 = word_tokenize(text5)

# Create an NLTK Text object
text_nltk = nltk.Text(words_text5)

# Find collocations
text_nltk.collocations(num=5)


# 8. DEFINE THE TWO LISTS PHRASE1, PHRASE2, EACH CONTAINING A FEW WORDS. JOIN THEM TOGETHER TO FORM A SENTENCE

# In[15]:


import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import webtext

# Download necessary datasets
nltk.download('punkt')
nltk.download('webtext')


# In[16]:


# Load the text of the Chat Corpus (text5)
text5 = webtext.raw('singles.txt')

# Tokenize the text into words
words_text5 = word_tokenize(text5)

# Create an NLTK Text object
text_nltk = nltk.Text(words_text5)

# Find collocations
text_nltk.collocations(num=5)

# Define two lists of words
phrase1 = ["Hello", "world"]
phrase2 = ["this", "is", "Python"]

# Join them to form a sentence
sentence = " ".join(phrase1 + phrase2)
print(sentence)

