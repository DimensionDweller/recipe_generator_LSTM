# LSTM RNN for Generating Novel Recipes

This project presents a Long Short-Term Memory Recurrent Neural Network (LSTM RNN) trained on a large recipe dataset. The model can generate novel, creative recipes, demonstrating the power of RNNs for sequence generation tasks. 

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)

## Background

A Recurrent Neural Network (RNN) is a type of artificial neural network designed to recognize patterns in sequences of data, such as text, genomes, handwriting, or the spoken word. RNNs are particularly useful when you want to predict what comes next in a sequence or generate new sequences. However, they suffer from the so-called "vanishing gradient" problem, which makes it hard for them to learn and remember long-range dependencies in the data.

Long Short-Term Memory (LSTM) is a type of RNN that includes a "memory cell" that can maintain information in memory for long periods of time. As you can see below, a set of gates is used to control when information enters memory, when it's output, and when it's forgotten. This architecture allows LSTM to learn longer sequences and outperform traditional RNNs in tasks such as language modeling and sequence generation. 

(https://github.com/DimensionDweller/recipe_generator_LSTM/assets/75709283/80c57dcb-4069-49dc-bf7b-dc763f30e4c4)


This project uses an LSTM RNN to generate novel recipes. By training the LSTM on a large corpus of existing recipes, the model learns the pattern and structure of recipes, and it can then generate new recipes that have never been seen before.

## Project Description

The model is trained on a large dataset of recipes in JSON format. Each recipe includes information such as the title, ingredients, and cooking instructions. The training was performed for 50 epochs.

The LSTM model is designed to generate text character by character. After the model is trained, you can give it a "seed" recipe title (e.g., "Chicken Soup") and it will generate a complete recipe that starts with that title.

## Model Architecture

The architecture of our LSTM RNN model is crucial to its performance. It consists of an embedding layer, an LSTM layer, and a linear layer:

### Embedding Layer
The embedding layer is used to convert the input words to dense vectors of fixed size. This is more efficient than one-hot encoding, and it allows the model to learn meaningful representations for the words. 

### LSTM Layer
The LSTM layer is the core of the model. It receives the word vectors from the embedding layer and outputs a new sequence of encoded states. The LSTM has a hidden state that's passed from one step in the sequence to the next, and it's this hidden state that it uses to "remember" information about the sequence.

### Linear Layer
The linear layer is used to decode the LSTM's output states into predicted words. It's a fully connected layer that converts the LSTM output to a distribution over the vocabulary.

The detailed architecture is as follows:

```
LSTMModel(
  (embedding): Embedding(20000, 128)
  (lstm): LSTM(128, 256, batch_first=True)
  (fc): Linear(in_features=256, out_features=20000, bias=True)
  (log_softmax): LogSoftmax(dim=2)
)
```

## Results

The LSTM RNN model was able to generate novel recipes that are coherent and follow the structure of a recipe. Here are a few examples of generated recipes:

```
Prompt: "recipe for roasted vegetables:"
Response: chop 1 / 2 cup onion and reserve for another use . place garlic in a large , heavy - bottomed saucepan over medium - high heat . add onions and sauté until tender , about 5 minutes . add garlic and stir - fry until water evaporates , about 2 minutes . add wine and bring to boil . reduce heat to medium and simmer , covered , until vegetables are tender , about 20 minutes . uncover and cook until liquid is reduced by half , about 30 minutes . add wine and boil until reduced by half. 
```

---

## Usage

Firstly, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/YourUsername/LSTM_Recipe_Generator.git
```

Navigate to the directory of the project:

```bash
cd LSTM_Recipe_Generator
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

Run the following command to start training the model:

```bash
python main.py
```

You can adjust the hyperparameters of the model by modifying the
