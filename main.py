import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

import json
import re
import string

import zipfile

from LSTMModel import LSTMModel
from TextDataset import TextDataset
from TextGenerator import TextGenerator


# Constants
VOCAB_SIZE = 20000
BATCH_SIZE = 64
EMBEDDING_DIM = 128
N_UNITS = 256
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def pad_punctuation(s):
    s = re.sub(f"([{string.punctuation}])", r" \1 ", s)
    s = re.sub(" +", " ", s)
    return s


def yield_tokens(data_iter):
    for text_str in data_iter:
        yield tokenizer(text_str)


def collate_batch(batch):
    stoi = vocab.get_stoi()
    batch = [torch.tensor([stoi.get(token, stoi['<unk>']) for token in tokens]) for tokens in batch]
    batch = pad_sequence(batch, batch_first=True, padding_value=stoi['<pad>'])
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y


if __name__ == "__main__":
    # Load and preprocess the data
    with open("data/full_format_recipes.json") as json_data:
        recipe_data = json.load(json_data)

    # Filter the dataset
    filtered_data = [
        "Recipe for " + x["title"] + " | " + " ".join(x["directions"])
        for x in recipe_data
        if "title" in x
        and x["title"] is not None
        and "directions" in x
        and x["directions"] is not None
    ]

    # Pad the punctuation, to treat them as separate 'words'
    text_data = [pad_punctuation(x) for x in filtered_data]

    # Prepare the tokenizer and the vocab
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(text_data), max_tokens=VOCAB_SIZE, specials=['<pad>', '<unk>'])
    vocab.set_default_index(vocab['<unk>'])

    # Create the dataset and the data loader
    text_ds = TextDataset(text_data, tokenizer)
    text_loader = DataLoader(text_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)

    # Create the model
    model = LSTMModel(VOCAB_SIZE, EMBEDDING_DIM, N_UNITS).to(DEVICE)

    # Define the loss function and optimizer
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # Define the number of epochs
    EPOCHS = 50

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        for i, (x, y) in enumerate(text_loader):
            # Move the data to the device
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # Forward pass
            y_pred = model(x)

            # Compute the loss
            loss = loss_fn(y_pred.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()

            # Compute gradient norm and clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

            # Print loss every 100 batches
            if i % 100 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

        # Generate text at the end of each epoch
        text_generator = TextGenerator(vocab)
        generated_text = text_generator.generate(model, DEVICE, "recipe for", max_tokens=100, temperature=1.0)
        print(generated_text)
