import torch
import torch.nn as nn
from torch.nn import functional as F

# -------- Hyperparameters --------
batch_size = 32 # number of independent sequences we will process in parallel
context_length = 8 # maximum context length for predictions
max_iters = 5000 # maximum iterations
eval_interval = 300
learning_rate = 1e-3
eval_iters = 200
n_embd = 32

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

@torch.no_grad() # this tells pytorch that for everything that happens in the function below, we will not be calling .backward(), meaning that PyTorch can be a lot more efficient with its memory usage
def estimate_loss():
    """estimate train and val loss by averaging over many batches"""
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


class Head(nn.Module):
    """One head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) # (H, C)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

    def forward(self, x):
        B, T, C = x.shape
        q = self.query(x) # (B, T, H)
        k = self.key(x) # (B, T, H)
        _, _, H = q.shape

        # Compute attention scores / logits (affinities)
        wei = q @ k.transpose(-2, -1) * H**-0.5 # (B, T, H) @ (B, T, H).transpose(-2, -1) = (B, T, H) @ (B, H, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1)

        # perform weighted aggregation of the values
        v = self.value(x) # (B, T, H)
        out = wei @ v # (B, T, T) @ (B, T, H) --> (B, T, H)
        return out

class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return out
        

class FeedForward(nn.Module):
    """Simple feed forward linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # original Transformer FFN uses d_ff = 4 * d_model (2048 vs. 512); Vaswani et al. (2017), Section 3.3
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # the projection layer back into the residual pathway
        )
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """A single Transformer block: communication followed by computation."""

    def __init__(self, n_embd, n_head):
        """
        n_embd: embedding dimension
        n_head: number of attention heads we want
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x)) # Notes: (1) the "x + ..." gives us residual connections, (2) we're using the pre-norm formulation, which departs from the original paper's post-norm setup
        x = x + self.ffwd(self.ln2(x)) 
        return x


# -------- Simple Bigram Model --------
class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) 
        self.position_embedding_table = nn.Embedding(context_length, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            nn.LayerNorm(n_embd),
        )
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # idx and targets are both (B, T) tensors of integers
        tok_emb = self.token_embedding_table(idx) # shape (B, T, C) (note that C == n_embd in this case)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) (due to broadcasting)
        x = self.blocks(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)

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
            # crop idx to the last context_length tokens
            idx_cond = idx[:, -context_length:]
            
            # get the predictions
            logits, loss = self(idx_cond)
            
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
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
context = idx=torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))