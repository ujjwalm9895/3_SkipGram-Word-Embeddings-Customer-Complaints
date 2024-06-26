# Import necessary libraries and modules
import os
import torch
import config  # Assumes a 'config.py' file is present
import argparse
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from Source.model import SkipGram
from Source.utils import load_file
from Source.data import SkipGramDataset

# Define a function for training the Skip-gram model
def train_sg(dataloader, model, criterion, optimizer, device, num_epochs):
    model.train()
    best_loss = 1e8
    patience = 0
    for i in range(num_epochs):
        epoch_loss = []
        print(f"Epoch {i+1} of {num_epochs}")
        for center_word, context_words in tqdm(dataloader):
            center_word = center_word.to(device)
            context_words = context_words.to(device)
            output, true_y = model(center_word, context_words)
            loss = criterion(output, true_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
        epoch_loss = np.mean(epoch_loss)
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience = 0
        else:
            patience += 1
        print(f"Loss: {epoch_loss}")
        if patience == 10:
            print("Early stopping...")
    model.save_files()

# Define the main function
def main(args_):
    # Determine the device to use (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load tokenized data from a file
    tokens = load_file(os.path.join(args_.output_path, args_.token_file))

    # Create a SkipGramDataset
    dataset = SkipGramDataset(input_data=tokens, context_window=args_.context_window,
                              out_path=args_.output_path, t=args_.t, k=args_.k)

    # Create a data loader for the SkipGramDataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args_.batch_size,
                                             shuffle=True, drop_last=True)

    # Initialize the Skip-gram model
    model = SkipGram(dataset.vocab_count, device, embedding_size=args_.embedding_size)
    
    # Move the model to the GPU if available
    if torch.cuda.is_available():
        model = model.cuda()

    # Define the loss function (Negative Log-Likelihood Loss)
    criterion = nn.NLLLoss()

    # Define the optimizer (Adam optimizer with learning rate 0.01)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Start training the Skip-gram model
    train_sg(dataloader, model, criterion, optimizer, device, args_.num_epochs)

# Entry point of the script
if __name__ == "__main__":
    # Create an argument parser to specify various hyperparameters and file paths
    parser = argparse.ArgumentParser()
    parser.add_argument("--token_file", type=str, default=config.token_file,
                        help="File containing word tokens")
    parser.add_argument("--output_path", type=str, default=config.output_folder,
                        help="Output folder name")
    parser.add_argument("--context_window", type=int, default=config.context_window,
                        help="Context window size")
    parser.add_argument("--t", type=float, default=config.t,
                        help="Threshold")
    parser.add_argument("--k", type=int, default=config.k,
                        help="Number of negative samples")
    parser.add_argument("--batch_size", type=int, default=config.batch_size,
                        help="Batch size of training")
    parser.add_argument("--embedding_size", type=int, default=config.embedding_size,
                        help="Embedding size of word vectors")
    parser.add_argument("--lr", type=float, default=config.lr,
                        help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=config.num_epochs,
                        help="Number of epochs")
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
