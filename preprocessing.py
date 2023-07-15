import re
import string
import zipfile
import json
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

def extract_zip_file(zip_file, extract_dir):
    # Open the zip file
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        # Extract all the contents of the zip file to the specified directory
        zip_ref.extractall(extract_dir)

def load_dataset(json_file):
    # Load the dataset
    with open(json_file) as json_data:
        data = json.load(json_data)
    return data

def filter_dataset(data):
    # Filter the dataset
    filtered_data = [
        "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
        for x in data
        if "title" in x
        and x["title"] is not None
        and "directions" in x
        and x["directions"] is not None
    ]
    return filtered_data

def pad_punctuation(s):
    # Pad the punctuation, to treat them as separate 'words'
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s

def build_vocab(text_data, tokenizer, max_tokens, specials):
    # Build the vocabulary
    def yield_tokens(data_iter):
        for text_str in data_iter:
            yield tokenizer(text_str)
    vocab = build_vocab_from_iterator(yield_tokens(text_data), max_tokens=max_tokens, specials=specials)
    vocab.set_default_index(vocab['<unk>'])
    return vocab
