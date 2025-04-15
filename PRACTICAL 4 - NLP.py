#!/usr/bin/env python
# coding: utf-8

# 1.	Explain the difference between assigning a list to a new variable using direct assignment (=) and using the copy() method. Provide code examples to illustrate the difference.
# 
# 2.	Write a function extract_nouns(text) that takes a text string as input and returns a list of all nouns in the text. Use NLTK's part-of-speech tagging for this task.
# 
# 3.	Demonstrate how to use list comprehension to create a list of the lengths of each word in a given sentence.
# 
# 4.	Write a function word_frequency(text) that takes a text string and returns a dictionary with words as keys and their frequencies as values.
# 
# 5.	Explain the concept of variable scope in Python with an example demonstrating the difference between local and global variables.

# 1.	Explain the difference between assigning a list to a new variable using direct assignment (=) and using the copy() method. Provide code examples to illustrate the difference.

# In[2]:


# direct assignment affects the original list, and how copy keeps it separate.

# Direct Assignment
list1 = [1, 2, 3]
list2 = list1  # list2 points to the same list as list1
list2.append(4)
print(f"list1: {list1}")  # Output: [1, 2, 3, 4]
print(f"list2: {list2}")  # Output: [1, 2, 3, 4]

# Using the copy() method
list3 = [1, 2, 3]
list4 = list3.copy()  # list4 is a new independent copy of list3
list4.append(4)
print(f"list3: {list3}")  # Output: [1, 2, 3]
print(f"list4: {list4}")  # Output: [1, 2, 3, 4]


# 2.	Write a function extract_nouns(text) that takes a text string as input and returns a list of all nouns in the text. Use NLTK's part-of-speech tagging for this task.

# In[4]:


import nltk
from nltk import word_tokenize, pos_tag

# Download the required resource for English
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng') # Download the resource for english language

def extract_nouns(text):
    # Tokenize the text and tag parts of speech
    words = word_tokenize(text)
    tagged_words = pos_tag(words)

    # Extract nouns (both singular and plural)
    nouns = [word for word, tag in tagged_words if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
    return nouns

# Example usage
text = "The quick brown fox jumped over the lazy dog."
print(extract_nouns(text))


# 3.	Demonstrate how to use list comprehension to create a list of the lengths of each word in a given sentence.

# In[5]:


sentence = "The quick brown fox jumped over the lazy dog."
word_lengths = [len(word) for word in sentence.split()]
print(word_lengths)


# 4.	Write a function word_frequency(text) that takes a text string and returns a dictionary with words as keys and their frequencies as values.

# In[6]:


import nltk
from nltk import word_tokenize

nltk.download('punkt')

def word_frequency(text):
    words = word_tokenize(text)
    frequency_dict = {}
    for word in words:
        word = word.lower()  # Normalize to lowercase
        frequency_dict[word] = frequency_dict.get(word, 0) + 1
    return frequency_dict

# Example usage
text = "This is a test. This is only a test."
print(word_frequency(text))


# 5.	Explain the concept of variable scope in Python with an example demonstrating the difference between local and global variables.

# In[7]:


#  Explanation:
# A global variable is defined outside any function and is accessible inside functions (if not overridden).
# A local variable is defined inside a function and only accessible within that function.

x = 10  # Global variable

def example_function():
    y = 20  # Local variable
    print(f"Local variable y: {y}")
    print(f"Global variable x: {x}")

example_function()


print(f"Global variable x outside function: {x}")

