"""
Skip-gram (Word2Vec) from scratch in PyTorch.

Idea: given a CENTER word, predict each word in its surrounding
context window. The learned embedding matrix becomes the word vectors.

Requirements: pip install torch matplotlib tqdm
"""

import re
from collections import Counter

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
CONTEXT_SIZE = 2        # words to the left/right of the center word
EMBEDDING_DIM = 24
BATCH_SIZE = 64
LEARNING_RATE = 0.01
NUM_EPOCHS = 300
SEED = 42

torch.manual_seed(SEED)

# ---------------------------------------------------------
# 1. Toy corpus (one sentence per line)
# ---------------------------------------------------------
TOY_DATA = """
the tall man walks to the tall building
the tall woman sees the tall tree
the short man walks to the short wall
the short woman sees the short bush
a tall tree grows next to a tall building
a short bush grows next to a short wall
the man and the woman walk to the tree
the old man sees the old building
the young woman sees the young tree
"""

# ---------------------------------------------------------
# 2. Tokenization and vocabulary
# ---------------------------------------------------------
UNK = "<unk>"


def tokenize(text):
    """Lowercase and split into word tokens."""
    return re.findall(r"[a-z0-9']+", text.lower())


def build_vocab(sentences):
    """Return (word_to_id, id_to_word). Index 0 is reserved for <unk>."""
    counts = Counter(word for sent in sentences for word in sent)
    id_to_word = [UNK] + [w for w, _ in counts.most_common()]
    word_to_id = {w: i for i, w in enumerate(id_to_word)}
    return word_to_id, id_to_word


# Keep sentences separate so context windows never cross sentence boundaries.
sentences = [tokenize(line) for line in TOY_DATA.splitlines() if line.strip()]
word_to_id, id_to_word = build_vocab(sentences)
vocab_size = len(id_to_word)

print(f"Sentences:       {len(sentences)}")
print(f"Vocabulary size: {vocab_size}")

# ---------------------------------------------------------
# 3. Build Skip-gram training pairs: (center_id, context_id)
# ---------------------------------------------------------
def make_skipgram_pairs(sentences, word_to_id, context_size):
    pairs = []
    for sentence in sentences:
        ids = [word_to_id.get(w, word_to_id[UNK]) for w in sentence]
        for center_pos, center_id in enumerate(ids):
            start = max(0, center_pos - context_size)
            end = min(len(ids), center_pos + context_size + 1)
            for ctx_pos in range(start, end):
                if ctx_pos != center_pos:
                    pairs.append((center_id, ids[ctx_pos]))
    return pairs


pairs = make_skipgram_pairs(sentences, word_to_id, CONTEXT_SIZE)
print(f"Skip-gram pairs: {len(pairs)}")
print("Examples:", [(id_to_word[c], id_to_word[x]) for c, x in pairs[:6]])

# Pairs are already integer ids, so the DataLoader's default collate
# turns them into two tensors: (center_ids, context_ids).
dataloader = DataLoader(pairs, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------
# 4. Model
# ---------------------------------------------------------
class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # word vectors
        self.output = nn.Linear(embedding_dim, vocab_size)         # scores over vocab

    def forward(self, center_ids):
        return self.output(self.embeddings(center_ids))  # (batch, vocab_size)


# ---------------------------------------------------------
# 5. Training
# ---------------------------------------------------------
def train(model, loader, criterion, optimizer, num_epochs):
    losses = []
    for epoch in tqdm(range(num_epochs), desc="Training"):
        model.train()
        total_loss = 0.0

        for center_ids, context_ids in loader:
            optimizer.zero_grad()
            logits = model(center_ids)
            loss = criterion(logits, context_ids)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        losses.append(total_loss / len(loader))

        if (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch + 1}/{num_epochs}  loss: {losses[-1]:.4f}")
    return losses


model = SkipGram(vocab_size, EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = train(model, dataloader, criterion, optimizer, NUM_EPOCHS)

plt.figure(figsize=(8, 4))
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Average loss")
plt.title("Skip-gram training loss")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# 6. Inference helpers
# ---------------------------------------------------------
def predict_context_words(word, top_k=5):
    """Most likely words to appear near `word`."""
    word = word.lower()
    if word not in word_to_id:
        print(f"'{word}' is not in the vocabulary.")
        return []

    model.eval()
    with torch.no_grad():
        probs = F.softmax(model(torch.tensor([word_to_id[word]])), dim=1)[0]
        probs[word_to_id[word]] = 0.0     # ignore the word itself
        probs[word_to_id[UNK]] = 0.0      # ignore <unk>
        scores, idx = torch.topk(probs, k=min(top_k, vocab_size - 2))

    return [(id_to_word[i.item()], s.item()) for s, i in zip(scores, idx)]


def find_similar_words(word, top_k=5):
    """Words whose embeddings are closest (cosine similarity) to `word`."""
    word = word.lower()
    if word not in word_to_id:
        print(f"'{word}' is not in the vocabulary.")
        return []

    model.eval()
    with torch.no_grad():
        emb = model.embeddings.weight
        sims = F.cosine_similarity(emb[word_to_id[word]].unsqueeze(0), emb, dim=1)
        sims[word_to_id[word]] = -1.0
        sims[word_to_id[UNK]] = -1.0
        scores, idx = torch.topk(sims, k=min(top_k, vocab_size - 2))

    return [(id_to_word[i.item()], s.item()) for s, i in zip(scores, idx)]


# ---------------------------------------------------------
# 7. Try it out
# ---------------------------------------------------------
test_word = "tall"

print(f"\nLikely context words for '{test_word}':")
for w, p in predict_context_words(test_word):
    print(f"  {w:<10} {p:.4f}")

print(f"\nWords with embeddings similar to '{test_word}':")
for w, s in find_similar_words(test_word):
    print(f"  {w:<10} {s:.4f}")

# ---------------------------------------------------------
# 8. Save model + vocabulary
# ---------------------------------------------------------
torch.save(
    {
        "model_state_dict": model.state_dict(),
        "id_to_word": id_to_word,
        "embedding_dim": EMBEDDING_DIM,
        "context_size": CONTEXT_SIZE,
    },
    "skipgram_model.pth",
)
print("\nSaved model to skipgram_model.pth")