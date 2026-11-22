# ============================================================
# LAB 3 - TEXT CLASSIFICATION WITH PYTORCH + TORCHTEXT
# ============================================================

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from torchtext.data.functional import to_map_style_dataset
from torchtext.vocab import build_vocab_from_iterator

from tqdm import tqdm


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# ============================================================
# 2. LOAD AG_NEWS DATASET
# ============================================================

train_iter = AG_NEWS(split="train")

test_iter = AG_NEWS(split="test")

print("AG_NEWS dataset loaded.")


# ============================================================
# 3. TOKENIZER
# ============================================================

tokenizer = get_tokenizer("basic_english")


# Test tokenizer
sample_text = "The company reported strong financial results."

print("\nOriginal text:")
print(sample_text)

print("\nTokens:")
print(tokenizer(sample_text))


# ============================================================
# 4. BUILD VOCABULARY
# ============================================================

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text.lower())


vocab = build_vocab_from_iterator(
    yield_tokens(train_iter),
    specials=["<unk>"]
)

# Unknown words get <unk> index
vocab.set_default_index(vocab["<unk>"])


print("\nVocabulary size:")
print(len(vocab))

print("\nExample vocabulary entries:")
print(vocab.get_itos()[:20])


# ============================================================
# 5. TEXT PIPELINE
# ============================================================

def text_pipeline(x):
    return vocab(tokenizer(x))


# Example
example = "The football team won the match."

print("\nExample text:")
print(example)

print("\nToken IDs:")
print(text_pipeline(example))


# ============================================================
# 6. LABEL PIPELINE
# ============================================================

def label_pipeline(x):
    return int(x) - 1


print("\nLabel mapping:")
print("1 ->", label_pipeline(1), "(World)")
print("2 ->", label_pipeline(2), "(Sports)")
print("3 ->", label_pipeline(3), "(Business)")
print("4 ->", label_pipeline(4), "(Sci/Tech)")


# ============================================================
# 7. CONVERT DATASETS TO MAP-STYLE DATASETS
# ============================================================

train_iter, test_iter = AG_NEWS()

train_dataset = to_map_style_dataset(train_iter)
test_dataset = to_map_style_dataset(test_iter)

print("\nTraining samples:", len(train_dataset))
print("Test samples:", len(test_dataset))


# ============================================================
# 8. TRAIN / VALIDATION SPLIT
# ============================================================

num_train = int(len(train_dataset) * 0.95)
num_valid = len(train_dataset) - num_train

split_train, split_valid = random_split(
    train_dataset,
    [num_train, num_valid]
)

print("\nTraining samples:", len(split_train))
print("Validation samples:", len(split_valid))


# ============================================================
# 9. COLLATE FUNCTION
# ============================================================

def collate_batch(batch):

    label_list = []
    text_list = []
    offsets = [0]

    for label, text in batch:

        # Convert label to zero-based class index
        label_list.append(label_pipeline(label))

        # Convert text into token IDs
        processed_text = torch.tensor(
            text_pipeline(text),
            dtype=torch.int64
        )

        text_list.append(processed_text)

        # Store length of this article
        offsets.append(processed_text.size(0))

    # Convert labels to tensor
    label_list = torch.tensor(
        label_list,
        dtype=torch.int64
    )

    # Calculate starting position of every article
    offsets = torch.tensor(
        offsets[:-1]
    ).cumsum(dim=0)

    # Concatenate all token IDs
    text_list = torch.cat(text_list)

    return (
        label_list.to(device),
        text_list.to(device),
        offsets.to(device)
    )


# ============================================================
# 10. CREATE DATALOADERS
# ============================================================

batch_size = 64

train_loader = DataLoader(
    split_train,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=collate_batch
)

valid_loader = DataLoader(
    split_valid,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch
)

print("\nDataLoaders created.")


# ============================================================
# 11. INSPECT ONE BATCH
# ============================================================

labels, text, offsets = next(iter(train_loader))

print("\nOne batch:")
print("Labels shape:", labels.shape)
print("Text shape:", text.shape)
print("Offsets shape:", offsets.shape)

print("\nFirst few labels:")
print(labels[:10])

print("\nFirst few token IDs:")
print(text[:20])

print("\nFirst few offsets:")
print(offsets[:10])


# ============================================================
# 12. TEXT CLASSIFICATION MODEL
# ============================================================

class TextClassificationModel(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class):

        super().__init__()

        # EmbeddingBag converts token IDs into one vector
        # for each document.
        self.embedding = nn.EmbeddingBag(
            vocab_size,
            embed_dim,
            sparse=False
        )

        # Classifier
        self.fc = nn.Linear(
            embed_dim,
            num_class
        )

        self.init_weights()


    def init_weights(self):

        initrange = 0.5

        self.embedding.weight.data.uniform_(
            -initrange,
            initrange
        )

        self.fc.weight.data.uniform_(
            -initrange,
            initrange
        )

        self.fc.bias.data.zero_()


    def forward(self, text, offsets):

        # Convert token IDs into document embeddings
        embedded = self.embedding(
            text,
            offsets
        )

        # Convert document embedding into
        # class scores
        return self.fc(embedded)


# ============================================================
# 13. CREATE MODEL
# ============================================================

num_class = 4
vocab_size = len(vocab)
embedding_dim = 64

model = TextClassificationModel(
    vocab_size,
    embedding_dim,
    num_class
).to(device)

print("\nModel:")
print(model)


# ============================================================
# 14. LOSS FUNCTION
# ============================================================

criterion = torch.nn.CrossEntropyLoss()


# ============================================================
# 15. OPTIMIZER
# ============================================================

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.1
)


# ============================================================
# 16. LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer,
    step_size=1,
    gamma=0.1
)


# ============================================================
# 17. TRAINING FUNCTION
# ============================================================

def train(model, data_loader):

    model.train()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    for labels, text, offsets in tqdm(
        data_loader,
        desc="Training",
        leave=False
    ):

        optimizer.zero_grad()

        # Forward pass
        output = model(
            text,
            offsets
        )

        # Calculate loss
        loss = criterion(
            output,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()

        # Predictions
        predictions = output.argmax(
            dim=1
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += labels.size(0)

    average_loss = total_loss / len(data_loader)

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# 18. EVALUATION FUNCTION
# ============================================================

def evaluate(model, data_loader):

    model.eval()

    total_loss = 0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():

        for labels, text, offsets in data_loader:

            output = model(
                text,
                offsets
            )

            loss = criterion(
                output,
                labels
            )

            total_loss += loss.item()

            predictions = output.argmax(
                dim=1
            )

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += labels.size(0)

    average_loss = total_loss / len(data_loader)

    accuracy = total_correct / total_samples

    return average_loss, accuracy


# ============================================================
# 19. TRAINING LOOP
# ============================================================

num_epochs = 10

best_valid_accuracy = 0

print("\nStarting training...\n")

for epoch in range(num_epochs):

    train_loss, train_accuracy = train(
        model,
        train_loader
    )

    valid_loss, valid_accuracy = evaluate(
        model,
        valid_loader
    )

    scheduler.step()

    print(
        f"Epoch {epoch + 1}/{num_epochs}"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: {valid_loss:.4f}"
    )

    print(
        f"Validation Accuracy: {valid_accuracy:.4f}"
    )

    print(
        f"Learning Rate: "
        f"{optimizer.param_groups[0]['lr']}"
    )

    print("-" * 50)

    # Save best model
    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = valid_accuracy

        torch.save(
            model.state_dict(),
            "my_model.pth"
        )

        print("Best model saved.")


# ============================================================
# 20. LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    torch.load(
        "my_model.pth",
        map_location=device
    )
)

print("\nBest model loaded.")


# ============================================================
# 21. TEST MODEL
# ============================================================

test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print("\nFinal Test Results")
print("------------------")

print(
    f"Test Loss: {test_loss:.4f}"
)

print(
    f"Test Accuracy: {test_accuracy:.4f}"
)


# ============================================================
# 22. CLASSIFY A SINGLE ARTICLE
# ============================================================

labels_map = {
    0: "World",
    1: "Sports",
    2: "Business",
    3: "Sci/Tech"
}


def predict(text):

    model.eval()

    token_ids = torch.tensor(
        text_pipeline(text),
        dtype=torch.int64
    )

    offsets = torch.tensor(
        [0],
        dtype=torch.int64
    )

    token_ids = token_ids.to(device)
    offsets = offsets.to(device)

    with torch.no_grad():

        output = model(
            token_ids,
            offsets
        )

        prediction = output.argmax(
            dim=1
        ).item()

    return labels_map[prediction]


# Example prediction

article = """
The football team won the championship after
a dramatic final match.
"""

print("\nExample classification:")
print(article)

print(
    "Predicted category:",
    predict(article)
)


# ============================================================
# 23. OPTIONAL T-SNE SECTION
# ============================================================

# This section requires scikit-learn and plotly.
#
# We are intentionally not installing those packages right now
# because they are not necessary for the core classification
# exercise.
#
# If you later want the visualization, we can add this section.


print("\nLab 3 completed.")