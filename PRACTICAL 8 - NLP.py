#!/usr/bin/env python
# coding: utf-8

# 1.	Write a Python program using NLTK to define a context-free grammar (CFG) that can parse simple sentences like "The cat sat on the mat." Use this grammar to generate the parse tree for the sentence.
# 
# 2.	Using NLTK, write a function that takes a sentence as input and returns all possible parse trees using a given CFG. Demonstrate this function with the sentence "I saw the man with the telescope."
# 
# 3.	Write a Python program using NLTK to create a recursive descent parser for a given CFG. Parse the sentence "She eats a sandwich." and display the parse tree.
# 
# 4.	Using NLTK, write a program to extract noun phrases from a sentence using a chunk grammar. Apply it to the sentence "The quick brown fox jumps over the lazy dog."
# 
# 5.	Write a Python function using NLTK that takes a sentence as input and returns the verb phrases (VP) using a chunk grammar. Demonstrate this function with the sentence "The cat is sleeping on the mat."

# 1.	Write a Python program using NLTK to define a context-free grammar (CFG) that can parse simple sentences like "The cat sat on the mat." Use this grammar to generate the parse tree for the sentence.

# In[1]:


import nltk

# Define a Context-Free Grammar (CFG)
cfg_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det N
    VP -> V PP | V
    PP -> P NP
    Det -> 'The' | 'the' | 'A' | 'a'
    N -> 'cat' | 'dog' | 'mat' | 'boy'
    V -> 'sat' | 'sleeps' | 'runs'
    P -> 'on' | 'under' | 'beside'
""")

# Initialize the parser
parser = nltk.ChartParser(cfg_grammar)

# Define the sentence to parse
sentence = "The cat sat on the mat".split()

# Generate and display the parse tree
for tree in parser.parse(sentence):
    print(tree)
    tree.pretty_print()  # Opens an NLTK window to visualize the tree


# 2.	Using NLTK, write a function that takes a sentence as input and returns all possible parse trees using a given CFG. Demonstrate this function with the sentence "I saw the man with the telescope."

# In[3]:


import nltk

# Define an ambiguous Context-Free Grammar (CFG)
cfg_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Det N | Det N PP | 'I'
    VP -> V NP | VP PP
    PP -> P NP
    Det -> 'the'
    N -> 'man' | 'telescope'
    V -> 'saw'
    P -> 'with'
""")

# Function to generate and return all possible parse trees
def generate_parse_trees(sentence):
    parser = nltk.ChartParser(cfg_grammar)
    sentence_tokens = sentence.split()
    trees = list(parser.parse(sentence_tokens))  # Generate all trees

    if not trees:
        print("No valid parse trees found.")
        return

    # Display all parse trees
    for i, tree in enumerate(trees, 1):
        print(f"Parse Tree {i}:")
        print(tree)
        tree.pretty_print()  # Opens NLTK tree visualization

# Test sentence
sentence = "I saw the man with the telescope"
generate_parse_trees(sentence)


# 3.	Write a Python program using NLTK to create a recursive descent parser for a given CFG. Parse the sentence "She eats a sandwich." and display the parse tree.

# In[5]:


import nltk

# Define the Context-Free Grammar (CFG)
cfg_grammar = nltk.CFG.fromstring("""
    S -> NP VP
    NP -> Pronoun | Det N
    VP -> V NP
    Det -> 'a' | 'the'
    N -> 'sandwich' | 'apple'
    Pronoun -> 'She' | 'He'
    V -> 'eats' | 'likes'
""")

# Function to parse a sentence using Recursive Descent Parser
def parse_sentence(sentence):
    parser = nltk.RecursiveDescentParser(cfg_grammar)
    sentence_tokens = sentence.split()  # Tokenize sentence
    trees = list(parser.parse(sentence_tokens))  # Generate parse trees

    if not trees:
        print("No valid parse trees found.")
        return

    # Display parse tree(s)
    for tree in trees:
        print(tree)
        tree.pretty_print()  # Opens the NLTK parse tree visualization

# Test sentence
sentence = "She eats a sandwich"
parse_sentence(sentence)


# 4.	Using NLTK, write a program to extract noun phrases from a sentence using a chunk grammar. Apply it to the sentence "The quick brown fox jumps over the lazy dog."

# In[9]:


import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

# Ensure necessary resources are downloaded
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

def extract_noun_phrases(sentence):
    # Tokenize and POS tag the sentence
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    # Define chunk grammar for Noun Phrases (NP)
    grammar = r"""
        NP: {<DT>?<JJ>*<NN>+}   # Determiner (optional) + Adjectives (0 or more) + Noun(s)
    """

    # Create a chunk parser
    chunk_parser = nltk.RegexpParser(grammar)

    # Apply chunking
    chunk_tree = chunk_parser.parse(pos_tags)

    # Extract and print noun phrases
    noun_phrases = []
    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'NP'):
        noun_phrases.append(" ".join(word for word, tag in subtree.leaves()))

    return noun_phrases

# Test sentence
sentence = "The quick brown fox jumps over the lazy dog."
noun_phrases = extract_noun_phrases(sentence)

# Output results
print("Extracted Noun Phrases:", noun_phrases)


# 5.	Write a Python function using NLTK that takes a sentence as input and returns the verb phrases (VP) using a chunk grammar. Demonstrate this function with the sentence "The cat is sleeping on the mat."

# In[12]:


import nltk

# Ensure necessary resources are downloaded
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def extract_verb_phrases(sentence):
    # Tokenize and POS tag the sentence
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)

    # Define chunk grammar for Verb Phrases (VP)
    grammar = r"""
        VP: {<VB.*><RB.*>?<VBG|VBN>?<PP>?}
    """
    # Explanation: VB.* (any verb), optional adverb (RB), optional gerund/past participle (VBG/VBN), optional prepositional phrase (PP)

    # Create a chunk parser
    chunk_parser = nltk.RegexpParser(grammar)

    # Apply chunking
    chunk_tree = chunk_parser.parse(pos_tags)

    # Extract and return verb phrases
    verb_phrases = []
    for subtree in chunk_tree.subtrees(filter=lambda t: t.label() == 'VP'):
        verb_phrases.append(" ".join(word for word, tag in subtree.leaves()))

    return verb_phrases

# Test sentence
sentence = "The cat is sleeping on the mat."
verb_phrases = extract_verb_phrases(sentence)

# Output results
print("Extracted Verb Phrases:", verb_phrases)

