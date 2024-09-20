import math
import os
import re
from collections import defaultdict
from typing import List, Tuple, Dict
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class VectorSpaceModel:
    def __init__(self):
        self.dictionary = {}
        self.postings = defaultdict(list)
        self.doc_lengths = {}
        self.N = 0  # Total number of documents
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()

    def tokenize(self, text: str) -> List[str]:
        # Case folding and normalization (lowercasing, removing non-alphanumeric characters)
        text = text.lower()
        tokens = nltk.word_tokenize(text)
        tokens = [re.sub(r'\W+', '', token) for token in tokens if token.isalnum()]

        # Remove stopwords and apply lemmatization
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]

        return tokens

    def index_documents(self, directory: str):
        self.N = 0
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                self.N += 1
                file_path = os.path.join(directory, filename)
                content = self.read_file_safe(file_path)
                if content:
                    self.index_document(filename, content)
        
        self.calculate_idf()
        self.normalize_document_vectors()

    def read_file_safe(self, file_path: str) -> str:
        encodings = ['utf-8', 'latin-1', 'ascii', 'utf-16']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        print(f"Warning: Unable to read file {file_path} with any of the attempted encodings.")
        return ""

    def index_document(self, doc_id: str, content: str):
        terms = self.tokenize(content)
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
        
        for term, freq in term_freq.items():
            if term not in self.dictionary:
                self.dictionary[term] = len(self.dictionary)
            self.postings[term].append((doc_id, freq))

    def calculate_idf(self):
        self.idf = {}
        for term, postings in self.postings.items():
            self.idf[term] = math.log10(self.N / len(postings))

    def normalize_document_vectors(self):
        for term, postings in self.postings.items():
            for i, (doc_id, freq) in enumerate(postings):
                tf = 1 + math.log10(freq)
                self.postings[term][i] = (doc_id, tf)
                self.doc_lengths[doc_id] = self.doc_lengths.get(doc_id, 0) + tf ** 2
        
        for doc_id in self.doc_lengths:
            self.doc_lengths[doc_id] = math.sqrt(self.doc_lengths[doc_id])

    def search(self, query: str) -> List[Tuple[str, float]]:
        query_terms = self.tokenize(query)
        query_vector = defaultdict(float)
        
        for term in query_terms:
            if term in self.dictionary:
                tf = 1 + math.log10(query_terms.count(term))
                query_vector[term] = tf * self.idf[term]
        
        query_length = math.sqrt(sum(w ** 2 for w in query_vector.values()))
        
        scores = defaultdict(float)
        for term, weight in query_vector.items():
            for doc_id, tf in self.postings[term]:
                scores[doc_id] += (weight * tf) / (query_length * self.doc_lengths[doc_id])
        
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:10]

# Usage
vsm = VectorSpaceModel()
vsm.index_documents("Corpus")

# Example search
query = "Developing your Zomato business account and profile is a great way to boost your restaurant's online reputation"
results = vsm.search(query)
for doc_id, score in results:
    print(f"{doc_id}: {score}")