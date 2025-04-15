#!/usr/bin/env python
# coding: utf-8

# 1.	Write a Python program using NLTK to perform part-of-speech tagging on the sentence: "The quick brown fox jumps over the lazy dog."
# 
# 2.	Using NLTK, write a function that takes a list of sentences and returns a list of part-of-speech tagged sentences.
# 
# 3.	Explain how to map the Penn Treebank POS tags to the Universal POS tags using NLTK. Provide a code example that tags a sentence and maps the tags accordingly.
# 
# 4.	Write a Python function using NLTK that takes a sentence as input and returns a list of all nouns in the sentence.
# 
# 5.	Using the Brown Corpus in NLTK, write a program to find the most common part-of-speech tag in the news category.

# 1.	Write a Python program using NLTK to perform part-of-speech tagging on the sentence: "The quick brown fox jumps over the lazy dog."

# In[1]:


import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sentence)
tagged = pos_tag(tokens)

print("POS Tagged Sentence:")
print(tagged)


# 2.	Using NLTK, write a function that takes a list of sentences and returns a list of part-of-speech tagged sentences.

# In[3]:


import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def tag_sentences(sentences):
    """
    Takes a list of sentences and returns a list where each sentence is POS tagged.
    """
    tagged_sentences = [pos_tag(word_tokenize(sentence)) for sentence in sentences]
    return tagged_sentences

# Example usage:
sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "NLTK is a powerful library for natural language processing."
]
tagged = tag_sentences(sentences)
print("Tagged sentences:")
for sent in tagged:
    print(sent)


# 3.	Explain how to map the Penn Treebank POS tags to the Universal POS tags using NLTK. Provide a code example that tags a sentence and maps the tags accordingly.

# In[6]:


import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset') # Added download for universal tagset

sentence = "The quick brown fox jumps over the lazy dog."
tokens = word_tokenize(sentence)
# Tag using Universal tagset
tagged_universal = pos_tag(tokens, tagset='universal')
print("POS-tagged sentence (Universal tagset):")
print(tagged_universal)


# 4.	Write a Python function using NLTK that takes a sentence as input and returns a list of all nouns in the sentence.

# In[7]:


import nltk
from nltk import word_tokenize, pos_tag

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

def extract_nouns(sentence):
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens, tagset='universal')
    # Filter for nouns; note: some proper nouns may be tagged as NOUN in the universal tagset
    nouns = [word for word, tag in tagged if tag == 'NOUN']
    return nouns

# Example usage:
sentence = "The quick brown fox jumps over the lazy dog."
nouns = extract_nouns(sentence)
print("Nouns in the sentence:")
print(nouns)


# 5.	Using the Brown Corpus in NLTK, write a program to find the most common part-of-speech tag in the news category.

# In[8]:


import nltk
from nltk.corpus import brown
from nltk import FreqDist

nltk.download('brown')

# Get tagged words from the 'news' category
tagged_words = brown.tagged_words(categories='news')
# Extract just the POS tags (the second element in each tuple)
tags = [tag for (word, tag) in tagged_words]

# Compute frequency distribution of tags
freq_dist = FreqDist(tags)
most_common_tag = freq_dist.most_common(1)[0]

print("Most common POS tag in the Brown 'news' category:")
print(most_common_tag)

