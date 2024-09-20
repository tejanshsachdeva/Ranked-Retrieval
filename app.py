import os
import math
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Download stopwords if necessary
nltk.download('stopwords')

# Initialize stop words and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Tokenization with stop word removal and stemming
def tokenize(text):
    tokens = re.findall(r'\b\w+\b', text.lower())  # Lowercase and split on word boundaries
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return tokens

index_cache = None

# Build index with caching
def build_index_if_needed(corpus_path):
    global index_cache
    if index_cache is None:
        index_cache = build_index(corpus_path)
    return index_cache

# Step 1: Parse the corpus and build dictionary + postings list
def build_index(corpus_path):
    dictionary = defaultdict(lambda: {'df': 0, 'postings': []})
    doc_lengths = {}
    doc_id_to_filename = {}
    N = 0
    
    for doc_id, filename in enumerate(os.listdir(corpus_path)):
        if filename.endswith(".txt"):
            try:
                N += 1
                doc_id_to_filename[doc_id] = filename
                filepath = os.path.join(corpus_path, filename)
                with open(filepath, 'r', encoding='utf-8') as file:
                    tokens = tokenize(file.read())
                    term_freqs = defaultdict(int)
                    for term in tokens:
                        term_freqs[term] += 1
                    length = 0
                    for term, freq in term_freqs.items():
                        log_tf = 1 + math.log10(freq)
                        length += log_tf ** 2
                        dictionary[term]['df'] += 1
                        dictionary[term]['postings'].append((doc_id, freq))
                    doc_lengths[doc_id] = math.sqrt(length)
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    
    return dictionary, doc_lengths, N, doc_id_to_filename

# Step 2: Process the query and compute tf-idf scores
def process_query(query, dictionary, N):
    tokens = tokenize(query)
    term_freqs = defaultdict(int)
    for term in tokens:
        term_freqs[term] += 1
    
    query_weights = {}
    for term, freq in term_freqs.items():
        if term in dictionary:
            df = dictionary[term]['df']
            idf = math.log10(N / df)
            log_tf = 1 + math.log10(freq)
            query_weights[term] = log_tf * idf
    return normalize_query_weights(query_weights)

# Normalize query weights
def normalize_query_weights(query_weights):
    norm = math.sqrt(sum(weight**2 for weight in query_weights.values()))
    return {term: weight / norm for term, weight in query_weights.items()}

# Step 3: Rank documents based on cosine similarity
def rank_documents(query_weights, dictionary, doc_lengths):
    scores = defaultdict(float)
    
    for term, weight in query_weights.items():
        if term in dictionary:
            for doc_id, freq in dictionary[term]['postings']:
                log_tf = 1 + math.log10(freq)
                scores[doc_id] += log_tf * weight
    
    for doc_id in scores:
        scores[doc_id] /= doc_lengths[doc_id]  # Normalize by document length
    
    ranked_docs = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked_docs[:10]

# Main entry point for searching
def search(query, corpus_path):
    if not query.strip():
        return []  # Return an empty list for empty queries
    
    dictionary, doc_lengths, N, doc_id_to_filename = build_index_if_needed(corpus_path)
    query_weights = process_query(query, dictionary, N)
    
    if not query_weights:  # Handle case where no terms in the query are in the dictionary
        return []
    
    ranked_docs = rank_documents(query_weights, dictionary, doc_lengths)
    
    results_with_filenames = [(doc_id_to_filename[doc_id], score) for doc_id, score in ranked_docs]
    
    return results_with_filenames


# Example usage
if __name__ == '__main__':
    corpus_path = 'corpus'  # Directory containing .txt files
    
    # Test Case 1
    query1 = "Developing your Zomato business account and profile is a great way to boost your restaurant’s online reputation"
    results1 = search(query1, corpus_path)
    print("Test Case 1 Results:", results1)

    # Test Case 2
    query2 = "Warwickshire, came from an ancient family and was the heiress to some land"
    results2 = search(query2, corpus_path)
    print("Test Case 2 Results:", results2)
