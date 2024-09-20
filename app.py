import os
import math
from collections import defaultdict

# Step 1: Parse the corpus and build dictionary + postings list
def build_index(corpus_path):
    dictionary = defaultdict(lambda: {'df': 0, 'postings': []})
    doc_lengths = {}
    doc_id_to_filename = {}
    N = 0  # Total number of documents
    
    for doc_id, filename in enumerate(os.listdir(corpus_path)):
        if filename.endswith(".txt"):
            N += 1
            doc_id_to_filename[doc_id] = filename  # Map doc_id to filename
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
    
    return dictionary, doc_lengths, N, doc_id_to_filename  # Return the mapping



# Step 2: Process the query and compute tf-idf scores
def process_query(query, dictionary, N):
    tokens = tokenize(query)
    query_weights = {}
    for term in tokens:
        if term in dictionary:
            df = dictionary[term]['df']
            idf = math.log10(N / df)
            tf = tokens.count(term)
            log_tf = 1 + math.log10(tf)
            query_weights[term] = log_tf * idf
    return query_weights

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

# Step 4: Utility function for tokenization
def tokenize(text):
    # Basic tokenization logic (e.g., lowercasing, splitting, removing punctuation)
    return text.lower().split()

# Main entry point for searching
def search(query, corpus_path):
    dictionary, doc_lengths, N, doc_id_to_filename = build_index(corpus_path)
    query_weights = process_query(query, dictionary, N)
    ranked_docs = rank_documents(query_weights, dictionary, doc_lengths)
    
    # Convert doc IDs to filenames for the final output
    results_with_filenames = [(doc_id_to_filename[doc_id], score) for doc_id, score in ranked_docs]
    
    return results_with_filenames

# Example usage
if __name__ == '__main__':
    corpus_path = 'corpus'  # Directory containing .txt files
    # query = "your search query here"
    # results = search(query, corpus_path)
    # print("Top 10 documents:", results)
    
    # Test Case 1
    query1 = "Developing your Zomato business account and profile is a great way to boost your restaurant’s online reputation"
    results1 = search(query1, corpus_path)
    print("Test Case 1 Results:", results1)

    # Test Case 2
    query2 = "Warwickshire, came from an ancient family and was the heiress to some land"
    results2 = search(query2, corpus_path)
    print("Test Case 2 Results:", results2)
