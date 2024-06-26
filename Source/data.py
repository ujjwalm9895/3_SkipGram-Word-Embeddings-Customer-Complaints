# Import necessary libraries and modules
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from Source.utils import save_file

# Define the SkipGramDataset class
class SkipGramDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, context_window=5, out_path="Output", t=1e-5, k=10):
        # Initialize the dataset with input data and other hyperparameters
        self.k = k
        self.context_window = context_window

        # Count word tokens in the input data
        print("Counting word tokens...")
        counter = Counter([t for d in tqdm(input_data) for t in d])
        self.vocab_count = len(counter)
        print(f"Unique words in the corpus: {self.vocab_count}")

        # Create positive data samples
        print("Creating data samples...")
        self.samples = self.positive_samples(input_data)

        # Generate word-to-index and index-to-word mappings
        word2idx = dict()
        idx2word = dict()
        sampling_prob = []
        print("Generating vocabulary...")
        for i, c in enumerate(counter.most_common(len(counter))):
            word2idx[c[0]] = i
            idx2word[i] = c[0]
            sampling_prob.append(c[1])
        self.word2idx = word2idx
        self.idx2word = idx2word

        # Calculate sampling probabilities for negative sampling
        print("Calculating sampling probabilities...")
        sampling_prob = np.sqrt(t / np.array(sampling_prob))
        sampling_prob = sampling_prob / np.sum(sampling_prob)
        self.sampling_prob = sampling_prob

        # Save word-to-index and index-to-word mappings to files
        print("Saving files...")
        self.save_files(out_path)

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, idx):
        # Get negative samples and return a data sample
        neg_words = self.negative_samples()
        center_word = self.word2idx[self.samples.loc[idx, "center_word"]]
        context_word = self.word2idx[self.samples.loc[idx, "context_word"]]
        return torch.tensor(center_word), torch.tensor([context_word] + neg_words)

    def positive_samples(self, input_data):
        # Create positive data samples
        samples = []
        cw = self.context_window
        for data in tqdm(input_data):
            text = [None] * cw + data + [None] * cw
            for i in range(cw, len(text) - cw):
                samples.append((text[i], text[i - cw:i] + text[i + 1: i + cw + 1]))
        samples = pd.DataFrame(samples, columns=["center_word", "context_word"])
        samples = samples.explode("context_word")
        samples.dropna(inplace=True)
        samples.reset_index(drop=True, inplace=True)
        return samples

    def negative_samples(self):
        # Generate negative samples for negative sampling
        neg_words = list(np.random.choice(np.arange(self.vocab_count), self.k, p=self.sampling_prob))
        return neg_words

    def save_files(self, out_path="Output"):
        # Save word-to-index and index-to-word mappings to files
        save_file(os.path.join(out_path, "word2idx.pkl"), self.word2idx)
        save_file(os.path.join(out_path, "idx2word.pkl"), self.idx2word)
