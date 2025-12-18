import numpy as np
from gensim.models import Word2Vec, KeyedVectors
import os

class ArabicWordEmbedder:
    '''
    Class to handle loading and retrieving Arabic word embeddings from AraVec.
    '''
    
    def __init__(self, model_path=None, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.model = None
        self.unknown_vector = None
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            print(f"Warning: Model path not found: {model_path}")
            print("Initializing with random embeddings")
            self.unknown_vector = np.zeros(embedding_dim, dtype=np.float32)
    
    def load_model(self, model_path):
        '''
        Loading pre-trained AraVec model from the specified path.
        '''
        print(f"Loading AraVec model from: {model_path}")

        try:
            lower = model_path.lower()

            if lower.endswith(".bin"):
                self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

            elif lower.endswith(".vec"):
                self.model = KeyedVectors.load_word2vec_format(model_path, binary=False)

            else:
                # .mdl format
                folder = os.path.dirname(model_path)
                base = os.path.basename(model_path)
                expected1 = os.path.join(folder, base + ".wv.vectors.npy")
                expected2 = os.path.join(folder, base + ".trainables.syn1neg.npy")
                if not (os.path.exists(expected1) and os.path.exists(expected2)):
                    print("Warning: AraVec .mdl often needs these companion files (.npy) in the same folder:")
                    print(" -", expected1)
                    print(" -", expected2)

                word2vec = Word2Vec.load(model_path)
                self.model = word2vec.wv

            self.embedding_dim = self.model.vector_size
            self.unknown_vector = np.mean(self.model.vectors, axis=0)

            print("Model loaded successfully!")
            print(f"  Model vocabulary size: {len(self.model)}")
            print(f"  Model embedding dimension: {self.embedding_dim}")

        except Exception as exception:
            print(f"Error loading model: {exception}")
            print("Falling back to zeros for unknown vectors.")
            self.model = None
            self.unknown_vector = np.zeros(self.embedding_dim, dtype=np.float32)

    
    def get_word_embedding(self, word):
        '''
        Get the embedding vector for a given word.
        '''
        if self.model is None:
            return self.unknown_vector.copy()
        
        try:
            return self.model[word]
        except KeyError:
            return self.unknown_vector.copy()
    

    def get_sequence_embeddings(self, letters, word_indices):
        '''
        Get the word embeddings for each character position in a sequence.
        Inputs:
            letters: list of characters
            word_indices: list of word indices per character
        Outputs:
            embeddings: numpy array of shape (sequence_length, embedding_dim)
        '''
        seq_len = len(letters)
        embeddings = np.zeros((seq_len, self.embedding_dim), dtype=np.float32)
        
        # build words from letters and word indices
        words = {}  # word_idx -> word_string
        for char, word_index in zip(letters, word_indices):
            if word_index == -1: # space character
                continue
            if word_index not in words:
                words[word_index] = "" # new word
            words[word_index] += char
        
        word_embeddings = {}
        for word_index, word in words.items():
            word_embeddings[word_index] = self.get_word_embedding(word)
        
        for char_index, word_index in enumerate(word_indices):
            if word_index == -1:
                embeddings[char_index] = self.unknown_vector # space characters get default (unknown) vector
            else:
                embeddings[char_index] = word_embeddings[word_index]
        
        return embeddings
    

    def get_batch_embeddings(self, all_letters, all_word_indices):
        '''
        Get word embeddings for a batch of sequences.
        Inputs:
            all_letters: list of letter sequences
            all_word_indices: list of word index sequences
        Outputs:
            list of numpy arrays, one per sequence'''
        return [
            self.get_sequence_embeddings(letters, word_indices)
            for letters, word_indices in zip(all_letters, all_word_indices)
        ]
