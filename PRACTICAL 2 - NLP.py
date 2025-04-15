#!/usr/bin/env python
# coding: utf-8

# 1.	(A)  Use the inaugural address corpus to find the total number of words and the total number of unique words in the inaugural addresses delivered in the 21st century.
# 
# 1. (B) FIRST 20 WORDS IN INAUGURAL
# 
# 2. (A) Write a Python program to find the frequency distribution of the words "democracy", "freedom", "liberty", and "equality" in all inaugural addresses using NLTK. [WITHOUT GRAPH]
# 
# 2. (B) WITH GRAPH
# 
# 3. Write a Python program to display the 5 most common words in the text of "Sense and Sensibility" by Jane Austen using the Gutenberg Corpus.
# 
# 4.	Generate a conditional frequency distribution of modal verbs ('can', 'could', 'may', 'might', 'must', 'will') across the categories of the Brown Corpus
# 
# 5.	Write a Python program to identify the longest word in "Moby Dick" from the Gutenberg Corpus.
# 
# 6.	Using the Brown Corpus, calculate the frequency of the word "government" across all categories.
# 
# 7.	Write a Python program using the Reuters Corpus to find the number of documents categorized under "crude".

# 1.	(A) Use the inaugural address corpus to find the total number of words and the total number of unique words in the inaugural addresses delivered in the 21st century.

# In[16]:


import nltk
from nltk.corpus import inaugural
from nltk import word_tokenize

# Download required resources
nltk.download('inaugural')
nltk.download('punkt')

# Filter files from the 21st century
twenty_first_century_files = [fileid for fileid in inaugural.fileids() if int(fileid[:4]) >= 2001]

# Collect words
words = []
for fileid in twenty_first_century_files:
    words += word_tokenize(inaugural.raw(fileid))

# Total and unique word counts
total_words = len(words)
unique_words = len(set(words))

print("Total words:", total_words)
print("Unique words:", unique_words)


# 1. (B) FIRST 20 WORDS IN INAUGURAL

# In[18]:


import nltk
from nltk.corpus import inaugural
from nltk import word_tokenize

# Download necessary resources
nltk.download('inaugural')
nltk.download('punkt_tab')

# Access the 1861 inaugural address (Lincoln's address)
address_1861 = inaugural.raw('1861-Lincoln.txt')

# Tokenize the address into words
address_words = word_tokenize(address_1861)

# Get and print the first 20 words
print("First 20 words of the 1861 inaugural address:")
print(address_words[:20])


# 2. Write a Python program to find the frequency distribution of the words "democracy", "freedom", "liberty", and "equality" in all inaugural addresses using NLTK. [WITHOUT GRAPH]

# In[19]:


import nltk
from nltk.corpus import inaugural
from nltk import word_tokenize, FreqDist

# Download necessary resources
nltk.download('inaugural')
nltk.download('punkt')

# List of target words (we'll count in lowercase for case-insensitivity)
target_words = ['democracy', 'freedom', 'liberty', 'equality']

# Combine all inaugural addresses into one text and tokenize
all_inaug_text = " ".join(inaugural.raw(fid) for fid in inaugural.fileids())
tokens = word_tokenize(all_inaug_text.lower())

# Create frequency distribution
freq_dist = FreqDist(tokens)

print("Frequency Distribution for Selected Words in Inaugural Addresses:")
for word in target_words:
    print(f"{word}: {freq_dist[word]}")


# 2. WITH GRAPH

# In[20]:


import nltk
from nltk.corpus import inaugural
import matplotlib.pyplot as plt

# Download the inaugural corpus if needed
nltk.download('inaugural')

# Initialize a dictionary to hold counts for 'freedom' in each address
freedom_counts = {}

for fileid in inaugural.fileids():
    # Extract the year from the fileid (e.g., "1861-Lincoln.txt")
    year = int(fileid.split('-')[0])
    # Convert text to lowercase for case-insensitive counting
    text_lower = inaugural.raw(fileid).lower()
    freedom_counts[year] = text_lower.count("freedom")

# Prepare data for plotting: sort by year
years = sorted(freedom_counts.keys())
counts = [freedom_counts[year] for year in years]

# Plot the frequency distribution
plt.figure(figsize=(10, 5))
plt.plot(years, counts, marker='o')
plt.title("Frequency of 'freedom' in Inaugural Addresses Over the Years")
plt.xlabel("Year")
plt.ylabel("Frequency of 'freedom'")
plt.grid(True)
plt.show()


# 3. Write a Python program to display the 5 most common words in the text of "Sense and Sensibility" by Jane Austen using the Gutenberg Corpus.

# In[21]:


import nltk
from nltk.corpus import gutenberg, stopwords
from nltk import word_tokenize, FreqDist
import string

# Download necessary resources
nltk.download('gutenberg')
nltk.download('punkt')
nltk.download('stopwords')

# Load "Sense and Sensibility" by Jane Austen
text = gutenberg.raw('austen-sense.txt')

# Tokenize and convert to lowercase
tokens = word_tokenize(text.lower())

# Define stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuation = set(string.punctuation)

# Filter tokens: remove stopwords and punctuation, and only keep alphanumeric tokens
filtered_tokens = [token for token in tokens if token.isalnum() and token not in stop_words]

# Create frequency distribution and get the 5 most common words
freq_dist = FreqDist(filtered_tokens)
most_common = freq_dist.most_common(5)

print("5 Most Common Words in 'Sense and Sensibility' (without filler words/punctuation):")
for word, count in most_common:
    print(f"{word}: {count}")


# 4.	Generate a conditional frequency distribution of modal verbs ('can', 'could', 'may', 'might', 'must', 'will') across the categories of the Brown Corpus

# In[22]:


import nltk
from nltk.corpus import brown
from nltk import ConditionalFreqDist

# Download the Brown corpus if necessary
nltk.download('brown')

# Define the modal verbs (in lowercase)
modals = ['can', 'could', 'may', 'might', 'must', 'will']

# Create a Conditional Frequency Distribution: condition is category, event is modal word occurrence.
cfd = ConditionalFreqDist(
    (category, word.lower())
    for category in brown.categories()
    for word in brown.words(categories=category)
    if word.lower() in modals
)

# Display counts for each modal verb across categories
print("Conditional Frequency Distribution of Modal Verbs in the Brown Corpus:")
for category in brown.categories():
    print(f"\nCategory: {category}")
    for modal in modals:
        print(f"  {modal}: {cfd[category][modal]}")


# 5.	Write a Python program to identify the longest word in "Moby Dick" from the Gutenberg Corpus.

# In[23]:


import nltk
from nltk.corpus import gutenberg
from nltk import word_tokenize

# Download Gutenberg corpus and tokenizer if needed
nltk.download('gutenberg')
nltk.download('punkt')

# Load "Moby Dick" (file id: melville-moby_dick.txt)
text = gutenberg.raw('melville-moby_dick.txt')
tokens = word_tokenize(text)

# Identify the longest word by length (if multiple, the first encountered is returned)
longest_word = max(tokens, key=len)

print("The longest word in 'Moby Dick' is:",longest_word)


# 6.	Using the Brown Corpus, calculate the frequency of the word "government" across all categories.

# In[24]:


import nltk
from nltk.corpus import brown

# Download the Brown corpus if needed
nltk.download('brown')

# Count occurrences of "government" (case-insensitive) in the entire Brown corpus
all_words = [word.lower() for word in brown.words()]
government_count = all_words.count("government")

print("Frequency of the word 'government' in the Brown Corpus:",government_count)


# 7.	Write a Python program using the Reuters Corpus to find the number of documents categorized under "crude".

# In[25]:


import nltk
from nltk.corpus import reuters

# Download the Reuters corpus if necessary
nltk.download('reuters')

# Count the documents categorized under "crude"
docs_crude = reuters.fileids("crude")
print("Number of Reuters documents categorized under 'crude':",len(docs_crude))

