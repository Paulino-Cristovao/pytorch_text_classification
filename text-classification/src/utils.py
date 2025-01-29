def load_data(file_path):
    # Function to load data from a given file path
    import pandas as pd
    data = pd.read_csv(file_path)
    return data

def preprocess_text(text):
    # Function to preprocess text data
    import re
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = text.lower()  # Convert to lowercase
    return text

def calculate_accuracy(predictions, labels):
    # Function to calculate accuracy of predictions
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    accuracy = correct / total
    return accuracy