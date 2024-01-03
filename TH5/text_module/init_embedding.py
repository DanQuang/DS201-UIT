from text_module.count_vectorizer import CountVectorizer
from text_module.tokenizer import Tokenizer

def build_text_embbeding(config):
    if config['text_embedding']['type']=='count_vector':
        return CountVectorizer(config)
    if config['text_embedding']['type']=='tokenizer':
        return Tokenizer(config)