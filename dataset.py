from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

# The TextDataset class inherits from PyTorch's Dataset class.
# This class is used to provide an interface for accessing all the training or testing samples in your dataset.
# Any dataset that subclasses torch.utils.data.Dataset needs __len__ and __getitem__ methods.
class TextDataset(Dataset):
    def __init__(self, text_data, tokenizer):
        self.text_data = text_data  # The text data to process
        self.tokenizer = tokenizer  # The tokenizer to use
    
    # Fetch a data sample for a given key. Returns a processed portion of the text data.
    def __getitem__(self, idx):
        tokens = self.tokenizer(self.text_data[idx])
        tokens = tokens[:200]  # Clip to maximum length of 200
        return tokens
    
    # Returns the number of samples in the data
    def __len__(self):
        return len(self.text_data)

# Function to handle collation. This function processes data and returns a batch.
# It pads the sequences to the same length, separates inputs and targets, and converts words to their indices.
def collate_batch(batch, vocab):
    stoi = vocab.get_stoi()  # Get the string-to-index mapping
    # Convert words to indices
    batch = [torch.tensor([stoi.get(token, stoi['<unk>']) for token in tokens]) for tokens in batch]
    # Pad sequences and separate inputs and targets
    batch = pad_sequence(batch, batch_first=True, padding_value=stoi['<pad>'])
    x = batch[:, :-1]
    y = batch[:, 1:]
    return x, y
