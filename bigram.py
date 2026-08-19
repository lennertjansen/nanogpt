# Code ported over from gpt-dev.ipynb, that's why it's so ugly

from webbrowser import get
import torch
import torch.nn as nn
from torch.nn import functional as F

# -------- Hyperparameters --------
batch_size = 32 # number of independent sequences we will process in parallel
context_length = 8 # maximum context length for predictions
max_iters = 3000 # maximum iterations
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps" # for my apple silicon people
else:
    device = "cpu"
# --------


# -------- Download, read, tokenize, and split data --------
# set seed for reproducibility
torch.manual_seed(2026)

# Download the raw tiny shakespeare dataset
# !curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

# read to inspect
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique characters that occur in the text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}
encode = lambda s: [stoi[c] for c in s] # encoder: takes a string, outputs a list of corresponding integers 
decode = lambda l: ''.join([itos[i] for i in l]) # encoder: takes a list of integers, outputs a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) # encode entire dataset and store as a torch.Tensor
n = int(0.9*len(data)) # first 90% will be train, remaining 10% will be validation set
train_data = data[:n]
val_data = data[n:]

# Data loading helper
def get_batch(split):
    # generate a small batch of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_length, (batch_size, )) # random sample of batch_size indices from the data (offset by the max context length)
    x = torch.stack([data[i:i+context_length] for i in ix])
    y = torch.stack([data[i+1:i+context_length+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    """Flush old losses and estimate means on splits"""
    out = {}
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# -------- Simple Bigram Model --------
class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size) 

    def forward(self, idx, targets=None):
        # idx and targets are both (B, T) tensors of integers
        logits = self.token_embedding_table(idx) # shape (B, T, C) (batch size, time (context_length), channel dimension (vocab_size, in this case))

        if targets is None:
            loss = None
        else:
            # Need to do some reshaping because PyTorch expects certain ordering of dims:
            # https://docs.pytorch.org/docs/2.13/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device) # move model params to device

# --------

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Training loop
for iter in range(max_iters):

    # every eval_interval steps, evaluate loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True) # flush stale gradients
    loss.backward() # bp
    optimizer.step() # update step
    

# generate from the model
context = idx=torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))