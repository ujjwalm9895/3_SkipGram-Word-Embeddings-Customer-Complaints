# Import the pickle module
import pickle

# Define a function to save an object as a pickle file
def save_file(name, obj):
    """
    Function to save an object as a pickle file
    """
    with open(name, 'wb') as f:
        pickle.dump(obj, f)

# Define a function to load a pickle object
def load_file(name):
    """
    Function to load a pickle object
    """
    return pickle.load(open(name, "rb"))
