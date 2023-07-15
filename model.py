import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

# LSTMModel is a subclass of PyTorch's nn.Module that defines a basic LSTM-based model.
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        """
        Initialize the LSTMModel.

        Args:
        - vocab_size (int): The size of the vocabulary, i.e., the number of unique words in the dataset.
        - embedding_dim (int): The size of the word embeddings.
        - hidden_dim (int): The size of the hidden states in the LSTM.
        """
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x):
        """
        Define the forward pass of the LSTMModel.

        Args:
        - x (torch.Tensor): The input tensor of shape (batch_size, sequence_length).

        Returns:
        - output (torch.Tensor): The output tensor of shape (batch_size, sequence_length, vocab_size).
        """
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return self.log_softmax(x)


class TextGenerator:
    def __init__(self, vocab, top_k=10):
        """
        Initialize the TextGenerator.

        Args:
        - vocab (torchtext.vocab): The vocabulary.
        - top_k (int): The number of top choices to sample the next word from.
        """
        self.vocab = vocab
        self.top_k = top_k

    def sample_from(self, logits, temperature):
        """
        Sample the next word index from a distribution of word probabilities.

        Args:
        - logits (torch.Tensor): The logit outputs from the model of shape (vocab_size).
        - temperature (float): The temperature factor to control the randomness of predictions by scaling the logits before applying softmax.

        Returns:
        - next_word_idx (int): The index of the next word sampled from the probability distribution.
        """
        probs = F.softmax(logits / temperature, dim=-1).cpu().numpy()
        return np.random.choice(len(probs), p=probs)

    def generate(self, model, device, start_prompt, max_tokens, temperature):
        """
        Generate text given a start prompt.

        Args:
        - model (nn.Module): The trained model to use for text generation.
        - device (torch.device): The device type ('cpu' or 'cuda') on which the model is loaded.
        - start_prompt (str): The start prompt to use for text generation.
        - max_tokens (int): The maximum number of tokens to generate.
        - temperature (float): The temperature factor to control the randomness of predictions.

        Returns:
        - generated_text (str): The generated text.
        """
        model.eval()
        tokens = [self.vocab.get_stoi()[token] for token in start_prompt.split()]
        tokens = torch.LongTensor(tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(max_tokens):
                output = model(tokens)
                next_token_logits = output[0, -1, :]
                next_token = self.sample_from(next_token_logits, temperature)
                tokens = torch.cat([tokens, torch.LongTensor([[next_token]]).to(device)], dim=1)
        generated_text = ' '.join(self.vocab.get_itos()[token] for token in tokens[0])
        return generated_text
