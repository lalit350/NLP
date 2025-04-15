#!/usr/bin/env python
# coding: utf-8

# 1.	Write a Python program to download the text of "Pride and Prejudice" by Jane Austen from Project Gutenberg, tokenize the text, and display the first 10 tokens.
# 
# 2.	Using NLTK, write a function that takes a URL as input, fetches the raw text from the webpage, and returns the number of words in the text.
# 
# 3.	Explain how to remove HTML tags from a web page's content using Python and NLTK. Provide a code example that fetches a web page, removes HTML tags, and prints the cleaned text.
# 
# 4.	Write a Python program that reads a text file, tokenizes its content into sentences, and prints the number of sentences in the file
# 
# 5.	Using regular expressions in Python, write a function that takes a list of words and returns a list of words that end with 'ing'.

# 1.	Write a Python program to download the text of "Pride and Prejudice" by Jane Austen from Project Gutenberg, tokenize the text, and display the first 10 tokens.

# In[23]:


get_ipython().system('pip install requests')

import nltk
from nltk import word_tokenize
import requests

# Download tokenizer
nltk.download('punkt')

# Download text from Project Gutenberg using requests
url = "https://www.gutenberg.org/files/1342/1342-0.txt"
response = requests.get(url)
raw_text = response.text

# Tokenize the text
tokens = word_tokenize(raw_text)
print("First 10 tokens:")
print(tokens[:10])


# 2.	Using NLTK, write a function that takes a URL as input, fetches the raw text from the webpage, and returns the number of words in the text.

# In[24]:


import nltk
import requests
from nltk import word_tokenize

nltk.download('punkt')

def analyze_words_from_url(url):
    response = requests.get(url)
    raw_text = response.text
    tokens = word_tokenize(raw_text)
    words = [word.lower() for word in tokens if word.isalpha()]  # keep only words
    total_words = len(tokens)
    unique_words = len(set(words))
    return total_words, unique_words

# Example usage:
url = "https://www.gutenberg.org/files/1342/1342-0.txt"
total, unique = analyze_words_from_url(url)
print(f"Total tokens (including punctuation): {total}")
print(f"Unique words (alphabetic only): {unique}")


# 3.	Explain how to remove HTML tags from a web page's content using Python and NLTK. Provide a code example that fetches a web page, removes HTML tags, and prints the cleaned text.

# In[25]:


get_ipython().system('pip install beautifulsoup4')

import nltk
import requests
from bs4 import BeautifulSoup

nltk.download('punkt')

def fetch_and_clean(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')  # Remove HTML tags
    cleaned_text = soup.get_text()
    return cleaned_text

# Example usage:
url = "https://github.com/huggingface"
clean_text = fetch_and_clean(url)
print("Cleaned Text:")
print(clean_text[:1000])  # print first 1000 characters to avoid flooding the screen

tokens = nltk.word_tokenize(clean_text)
print("First 10 tokens:", tokens[:10])


# 4.	Write a Python program that reads a text file, tokenizes its content into sentences, and prints the number of sentences in the file

# In[26]:


import nltk
from nltk import sent_tokenize

# Download sentence tokenizer
nltk.download('punkt')

# Open and read the file
with open('para1.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# Tokenize into sentences
sentences = sent_tokenize(text)

# Output the number of sentences
print("Number of sentences in the file:", len(sentences))


# 5.	Using regular expressions in Python, write a function that takes a list of words and returns a list of words that end with 'ing'.

# In[27]:


import re

def words_ending_with_ing(word_list):
    pattern = re.compile(r'.*ing$')
    return [word for word in word_list if pattern.match(word)]

# Example usage:
words = ["running", "jog", "swimming", "play", "coding", "read"]
ing_words = words_ending_with_ing(words)
print("Words ending with 'ing':", ing_words)

