from tqdm import tqdm
import torch
from torch import optim
from torch.nn import NLLLoss
from torch.nn.utils import clip_grad_norm_
import wandb
from wandb.sdk.integration_utils.data_logging import Html
from dataset import TextDataset, collate_batch
from model import LSTMModel, TextGenerator
from preprocessing import build_vocab, extract_zip_file, load_dataset, filter_dataset, pad_punctuation
from torchtext.data.utils import get_tokenizer


def train(model, optimizer, text_loader, device, start_epoch, num_epochs, checkpoint_path=None, log_wandb=True):
    # Load the checkpoint if it exists
    if checkpoint_path is not None and log_wandb:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print('Loaded model from checkpoint')
        
    # Define the loss function
    loss_fn = NLLLoss()

    # Training loop
    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()  # Ensure the model is in training mode
        for i, (x, y) in enumerate(text_loader):
            # Move the data to the device
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            y_pred = model(x)

            # Compute the loss
            loss = loss_fn(y_pred.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            
            # Compute gradient norm and clip gradients
            grad_norm = clip_grad_norm_(model.parameters(), max_norm=1)
            
            optimizer.step()

            # Print loss every 100 batches
            if i % 100 == 0:
                print(f'Epoch {epoch}, Batch {i}, Loss: {loss.item()}')

            # Log loss and perplexity to Weights & Biases
            if log_wandb:
                wandb.log({
                    "Epoch": epoch,
                    "Batch": i,
                    "Loss": loss.item(),
                    "Perplexity": torch.exp(loss).item(),
                    "Gradient Norm": grad_norm.item()
                })

        text_generator = TextGenerator(vocab)
        generated_text = text_generator.generate(model, device, "recipe for", max_tokens=100, temperature=1.0)
        print(generated_text)
        if log_wandb:
            wandb.log({"Generated Text": Html(f"<pre>{generated_text}</pre>", inject=False)})

        # Save a checkpoint after each epoch
        if log_wandb:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, f'checkpoint_epoch_{epoch}.pth')

            # log the checkpoint to wandb
            wandb.save(f'checkpoint_epoch_{epoch}.pth')
