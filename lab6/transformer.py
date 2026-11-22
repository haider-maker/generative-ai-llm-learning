import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.nn.utils.rnn import pad_sequence
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =============================================================
# 1. TEXT PIPELINE
# =============================================================

# AG_NEWS labels are 1..4 -> we shift them to 0..3
LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

# Create iterators (each item is a tuple: (label, text))
train_iter = AG_NEWS(split="train")

# Tokenizer: splits a sentence into word tokens
tokenizer = get_tokenizer("basic_english")


def yield_tokens(data_iter):
    """Yield the token list of every article (used to build the vocab)."""
    for _, text in data_iter:
        yield tokenizer(text)


# Build the vocabulary from the training data
# <unk> = unknown word, <pad> = padding token
vocab = build_vocab_from_iterator(
    yield_tokens(train_iter), min_freq=5, specials=["<unk>", "<pad>"]
)
vocab.set_default_index(vocab["<unk>"])  # unseen words -> <unk>
PAD_IDX = vocab["<pad>"]

# Pipelines: raw text -> list of ids, raw label -> 0..3
text_pipeline = lambda text: vocab(tokenizer(text))
label_pipeline = lambda label: int(label) - 1

# =============================================================
# 2. DATASETS + COLLATE FUNCTION + DATALOADERS
# =============================================================

MAX_LEN = 128  # truncate long articles (also the size of our pos. encoding table)
BATCH_SIZE = 64


def collate_batch(batch):
    """Turn a list of (label, text) into padded tensors for one batch."""
    labels, texts = [], []
    for label, text in batch:
        labels.append(label_pipeline(label))
        ids = text_pipeline(text)[:MAX_LEN]            # truncate
        texts.append(torch.tensor(ids, dtype=torch.long))
    # Pad every sequence in the batch to the longest one -> (batch, seq_len)
    texts = pad_sequence(texts, batch_first=True, padding_value=PAD_IDX)
    labels = torch.tensor(labels, dtype=torch.long)
    return texts.to(device), labels.to(device)


# Convert iterators to map-style datasets (needed for random_split)
train_dataset = to_map_style_dataset(AG_NEWS(split="train"))
test_dataset = to_map_style_dataset(AG_NEWS(split="test"))

# Allocate 95% of train data for training, 5% for validation
num_train = int(len(train_dataset) * 0.95)
train_data, valid_data = random_split(
    train_dataset, [num_train, len(train_dataset) - num_train]
)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# =============================================================
# 3. MODEL
# =============================================================


class PositionalEncoding(nn.Module):
    """Adds sine/cosine position information to the embeddings."""

    def __init__(self, d_model, max_len=MAX_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)                       # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)                           # even dims
        pe[:, 1::2] = torch.cos(position * div_term)                           # odd dims
        self.register_buffer("pe", pe.unsqueeze(0))                            # (1, max_len, d_model)

    def forward(self, x):                                                      # x: (batch, seq, d_model)
        return x + self.pe[:, : x.size(1)]


class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes, d_model=128, nhead=4,
                 num_layers=2, dim_ff=256, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        # 1) Embedding layer (padding index stays a zero vector)
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=PAD_IDX)
        # 2) Positional encoding
        self.pos_encoding = PositionalEncoding(d_model)
        # 3) Transformer encoder (stack of encoder layers)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        # 4) Classifier layer
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):                                  # x: (batch, seq)
        pad_mask = (x == PAD_IDX)                          # True where token is padding
        x = self.embedding(x) * math.sqrt(self.d_model)    # scale embeddings
        x = self.pos_encoding(x)
        x = self.encoder(x, src_key_padding_mask=pad_mask) # (batch, seq, d_model)

        # Mean-pool over real tokens only (ignore padding)
        keep = (~pad_mask).unsqueeze(-1).float()           # (batch, seq, 1)
        x = (x * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1)

        return self.fc(self.dropout(x))                    # (batch, num_classes)


model = TransformerClassifier(len(vocab), num_classes=len(LABELS)).to(device)

# =============================================================
# 4. TRAINING
# =============================================================

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
EPOCHS = 5


def evaluate(loader):
    """Return accuracy on a dataloader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for texts, labels in loader:
            preds = model(texts).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0
    for texts, labels in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(texts), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # keeps training stable
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch} | loss: {total_loss / len(train_loader):.4f} "
          f"| val acc: {evaluate(valid_loader):.4f}")

print(f"Test accuracy: {evaluate(test_loader):.4f}")

# =============================================================
# 5. PREDICTION
# =============================================================


def predict(text):
    """Return the predicted category name for a raw article string."""
    model.eval()
    with torch.no_grad():
        ids = text_pipeline(text)[:MAX_LEN]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)  # add batch dim
        return LABELS[model(x).argmax(dim=1).item()]


article = """Canada navigated a stiff test against the Republic of Ireland on a rain soaked evening in Perth, coming from behind to claim a vital 2-1 victory at the Women’s World Cup.
Katie McCabe opened the scoring with an incredible Olimpico goal – scoring straight from a corner kick – as her corner flew straight over the despairing Canada goalkeeper Kailen Sheridan at Perth Rectangular Stadium in Australia.
Just when Ireland thought it had safely navigated itself to half time with a lead, Megan Connolly failed to get a clean connection on a clearance with the resulting contact squirming into her own net to level the score.
Minutes into the second half, Adriana Leon completed the turnaround for the Olympic champion, slotting home from the edge of the area to seal the three points."""

print("Predicted category:", predict(article))