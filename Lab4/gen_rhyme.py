import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer


# Load the trained model and vocabulary
checkpoint = torch.load(
    "ngram_model.pth",
    map_location="cpu",
    weights_only=True
)

N = checkpoint["N"]
embedding_dim = checkpoint["embedding_dim"]
vocab_tokens = checkpoint["vocab_tokens"]

token_to_index = {
    token: index for index, token in enumerate(vocab_tokens)
}
unknown_index = token_to_index["<unk>"]
tokenizer = get_tokenizer("basic_english")


# Recreate the trained model architecture
class NGramLanguageModeler(nn.Module):
    def __init__(self, vocab_size, embedding_dim, N):
        super().__init__()
        self.N = N
        self.embedding_dim = embedding_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(N * embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        embeds = embeds.reshape(-1, self.N * self.embedding_dim)
        out = F.relu(self.linear1(embeds))
        return self.linear2(out)


model = NGramLanguageModeler(
    len(vocab_tokens), embedding_dim, N
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


# Function 1: Select a random starting line
def pickrandomline(nursery_rhymes):
    lines = [
        line.strip()
        for line in nursery_rhymes.splitlines()
        if line.strip()
    ]
    return random.choice(lines)


# Function 2: Generate a rhyme using the trained model
def write_song(model, my_rhyme, number_of_words=100):
    generated_tokens = tokenizer(my_rhyme.lower())

    for _ in range(number_of_words):
        context_words = generated_tokens[-N:]

        context_ids = [
            token_to_index.get(word, unknown_index)
            for word in context_words
        ]

        context = torch.tensor([context_ids], dtype=torch.long)

        with torch.no_grad():
            predictions = model(context)
            word_idx = torch.argmax(predictions, dim=1).item()

        generated_tokens.append(vocab_tokens[word_idx])

    return " ".join(generated_tokens)


# Starting lines for generation
nursery_rhymes = """
Little Miss Muffet
Twinkle twinkle little star
Jack and Jill went up the hill
Mary had a little lamb
The wheels on the bus go round and round
"""

# Generate and print a new rhyme
selected_line = pickrandomline(nursery_rhymes)
generated_rhyme = write_song(model, selected_line)

print("Starting line:", selected_line)
print("\nGenerated nursery rhyme:")
print(generated_rhyme)