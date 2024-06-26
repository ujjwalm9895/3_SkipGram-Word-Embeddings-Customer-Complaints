# Number of topics (clusters) in the model
k = 10

# Learning rate for model training
lr = 0.01

# Threshold for convergence
t = 1e-5

# Number of training epochs
num_epochs = 50

# Batch size for training data
batch_size = 512

# Context window size for word embeddings
context_window = 5

# Dimensionality of word embeddings
embedding_size = 64

# Input folder path
input_folder = "Input"

# Output folder path
output_folder = "Output"

# File containing tokenized data
token_file = "tokens.pkl"

# Input CSV file name
file_name = "complaints.csv"

# Column name in the CSV file that contains text data
col_name = "Consumer complaint narrative"
