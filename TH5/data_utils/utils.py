import torch

def preprocess_sentence(sentence):
    return [word.lower().strip() for word in sentence]