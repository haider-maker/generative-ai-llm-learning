# ============================================================
# Lab 2: Comparative Text Tokenization and Performance Analysis
# ============================================================

# ------------------------------------------------------------
# Step 1: Import the required libraries
# ------------------------------------------------------------

import nltk
import spacy

from transformers import BertTokenizer, XLNetTokenizer
from collections import Counter
from datetime import datetime


# ------------------------------------------------------------
# Step 2: Download the NLTK tokenizer data
# ------------------------------------------------------------
# NLTK needs this resource to perform word tokenization.
# We use download() so that the required resource is available.

nltk.download("punkt")
nltk.download("punkt_tab")


# ------------------------------------------------------------
# Step 3: Load the spaCy English tokenizer
# ------------------------------------------------------------
# We only need tokenization here, so we don't need to
# download a complete language model such as en_core_web_sm.
#
# spacy.blank("en") gives us a basic English tokenizer.

nlp = spacy.blank("en")


# ------------------------------------------------------------
# Step 4: Load the BERT tokenizer
# ------------------------------------------------------------
# BERT uses WordPiece-based subword tokenization.
#
# The tokenizer will automatically download the required
# tokenizer files the first time it is used.

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")


# ------------------------------------------------------------
# Step 5: Load the XLNet tokenizer
# ------------------------------------------------------------
# XLNet uses a SentencePiece-based tokenizer.
#
# The required tokenizer files will be downloaded the first
# time the tokenizer is loaded.

xlnet_tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")


# ------------------------------------------------------------
# Step 6: Define the text that we want to tokenize
# ------------------------------------------------------------

text = """
Going through the world of tokenization has been like walking
through a huge maze made of words, symbols, and meanings.
Each turn shows a bit more about the cool ways computers learn
to understand our language. And while I'm still finding my way
through it, the journey’s been enlightening and, honestly, a
bunch of fun. Eager to see where this learning path takes me next!
"""


# ------------------------------------------------------------
# Step 7: Create a function to display token frequencies
# ------------------------------------------------------------
# Counter() counts how many times each token appears.
#
# Example:
# ["the", "world", "the"]
#
# becomes:
# {"the": 2, "world": 1}

def show_frequencies(tokens, method_name):

    frequencies = Counter(tokens)

    print(f"\n{method_name} Token Frequencies:")

    for token, count in frequencies.items():
        print(f"{token}: {count}")


# ------------------------------------------------------------
# Step 8: NLTK tokenization
# ------------------------------------------------------------
# NLTK performs traditional word/punctuation tokenization.
#
# We record the time immediately before tokenization and
# immediately after tokenization to measure processing time.

start_time = datetime.now()

nltk_tokens = nltk.word_tokenize(text)

end_time = datetime.now()

nltk_time = end_time - start_time


# Display NLTK results

print("\n" + "=" * 60)
print("NLTK TOKENIZATION")
print("=" * 60)

print("Tokens:")
print(nltk_tokens)

print(f"\nNumber of tokens: {len(nltk_tokens)}")
print(f"Processing time: {nltk_time}")

show_frequencies(nltk_tokens, "NLTK")


# ------------------------------------------------------------
# Step 9: spaCy tokenization
# ------------------------------------------------------------
# spaCy processes the text and creates a Doc object.
# Each token in the Doc can be accessed individually.

start_time = datetime.now()

doc = nlp(text)

spacy_tokens = [token.text for token in doc]

end_time = datetime.now()

spacy_time = end_time - start_time


# Display spaCy results

print("\n" + "=" * 60)
print("SPACY TOKENIZATION")
print("=" * 60)

print("Tokens:")
print(spacy_tokens)

print(f"\nNumber of tokens: {len(spacy_tokens)}")
print(f"Processing time: {spacy_time}")

show_frequencies(spacy_tokens, "spaCy")


# ------------------------------------------------------------
# Step 10: BERT tokenization
# ------------------------------------------------------------
# BERT uses WordPiece subword tokenization.
#
# Some words may remain as one token, while uncommon words
# can be divided into smaller subword pieces.
#
# add_special_tokens=False means we don't add special tokens
# such as [CLS] and [SEP] because we are only comparing
# tokenization here.

start_time = datetime.now()

bert_tokens = bert_tokenizer.tokenize(
    text,
    add_special_tokens=False
)

end_time = datetime.now()

bert_time = end_time - start_time


# Display BERT results

print("\n" + "=" * 60)
print("BERT TOKENIZATION")
print("=" * 60)

print("Tokens:")
print(bert_tokens)

print(f"\nNumber of tokens: {len(bert_tokens)}")
print(f"Processing time: {bert_time}")

show_frequencies(bert_tokens, "BERT")


# ------------------------------------------------------------
# Step 11: XLNet tokenization
# ------------------------------------------------------------
# XLNet uses SentencePiece-based subword tokenization.
#
# Again, we disable special tokens because our goal is to
# compare the tokenization itself.

start_time = datetime.now()

xlnet_tokens = xlnet_tokenizer.tokenize(
    text,
    add_special_tokens=False
)

end_time = datetime.now()

xlnet_time = end_time - start_time


# Display XLNet results

print("\n" + "=" * 60)
print("XLNET TOKENIZATION")
print("=" * 60)

print("Tokens:")
print(xlnet_tokens)

print(f"\nNumber of tokens: {len(xlnet_tokens)}")
print(f"Processing time: {xlnet_time}")

show_frequencies(xlnet_tokens, "XLNet")


# ------------------------------------------------------------
# Step 12: Final comparison
# ------------------------------------------------------------
# Here we summarize the number of tokens and processing time
# produced by each tokenizer.

print("\n" + "=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

print(f"NLTK   : {len(nltk_tokens):4d} tokens | Time: {nltk_time}")
print(f"spaCy  : {len(spacy_tokens):4d} tokens | Time: {spacy_time}")
print(f"BERT   : {len(bert_tokens):4d} tokens | Time: {bert_time}")
print(f"XLNet  : {len(xlnet_tokens):4d} tokens | Time: {xlnet_time}")