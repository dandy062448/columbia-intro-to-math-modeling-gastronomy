from pathlib import Path
from collections import Counter
import ast
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ----------------------------
# Settings
# ----------------------------

CSV_PATH = Path("RecipeNLG_dataset.csv")

MAX_RECIPES = 50_000
MIN_INGREDIENT_FREQ = 10
EMBEDDING_DIM = 50
EPOCHS = 5
BATCH_SIZE = 256
LEARNING_RATE = 0.001
TEST_FRACTION = 0.10
RANDOM_SEED = 46

MODEL_OUTPUT_PATH = "ingredient_cbow_model.pt"
VOCAB_OUTPUT_PATH = "ingredient_vocab.txt"

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ----------------------------
# Data cleaning
# ----------------------------

JUNK_TOKENS = {
    "",
    "or",
    "and",
    "of",
    "of sauce",
    "homemade",
    "for the muffins",
    "muffins",
    "cream cheese filling",
    "pack of ready",
    "ready",
    "none",
}


def parse_ner_cell(cell):
    """
    Convert the NER column from a string like:
    '["brown sugar", "milk", "vanilla"]'
    into a Python list.
    """
    if pd.isna(cell):
        return []

    try:
        items = ast.literal_eval(cell)
    except Exception:
        return []

    if not isinstance(items, list):
        return []

    return items


def clean_ingredient(token):
    """
    Basic ingredient cleaning.
    This is intentionally simple for the first working model.
    """
    if not isinstance(token, str):
        return None

    token = token.strip().lower()

    # Remove surrounding punctuation
    token = token.strip(" .,:;()[]{}'\"")

    # Normalize a few common phrases
    token = token.replace("freshly ground ", "")
    token = token.replace("extra-virgin ", "extra virgin ")
    token = token.replace("all-purpose", "all purpose")

    # Remove tokens with no letters
    if not any(ch.isalpha() for ch in token):
        return None

    if token in JUNK_TOKENS:
        return None

    if len(token) < 2:
        return None

    return token


def load_recipes(csv_path, max_recipes):
    """
    Load a limited number of recipes from RecipeNLG using only the NER column.
    Each recipe becomes a list of cleaned ingredient tokens.
    """
    print(f"Reading up to {max_recipes:,} recipes from {csv_path}...")

    df = pd.read_csv(csv_path, usecols=["NER"], nrows=max_recipes)

    recipes = []

    for cell in df["NER"]:
        raw_ingredients = parse_ner_cell(cell)

        cleaned = []
        for item in raw_ingredients:
            ingredient = clean_ingredient(item)
            if ingredient is not None:
                cleaned.append(ingredient)

        # Remove duplicates inside a recipe while preserving order
        cleaned = list(dict.fromkeys(cleaned))

        # Keep only recipes with enough context
        if len(cleaned) >= 3:
            recipes.append(cleaned)

    print(f"Usable recipes: {len(recipes):,}")
    return recipes


# ----------------------------
# Vocabulary
# ----------------------------

def build_vocabulary(recipes, min_freq):
    """
    Build ingredient vocabulary.
    Ingredient ID 0 is reserved for padding.
    """
    counter = Counter()

    for recipe in recipes:
        counter.update(recipe)

    vocab = ["<PAD>"]

    for ingredient, count in counter.most_common():
        if count >= min_freq:
            vocab.append(ingredient)

    ingredient_to_id = {ingredient: idx for idx, ingredient in enumerate(vocab)}
    id_to_ingredient = {idx: ingredient for ingredient, idx in ingredient_to_id.items()}

    print(f"Unique raw ingredients: {len(counter):,}")
    print(f"Vocabulary size after min frequency filter: {len(vocab):,}")

    with open(VOCAB_OUTPUT_PATH, "w", encoding="utf-8") as f:
        for ingredient in vocab:
            f.write(ingredient + "\n")

    print(f"Saved vocabulary to {VOCAB_OUTPUT_PATH}")

    return ingredient_to_id, id_to_ingredient


def recipes_to_id_lists(recipes, ingredient_to_id):
    """
    Convert ingredient strings into ingredient IDs.
    Ingredients not in the vocabulary are dropped.
    """
    recipe_ids = []

    for recipe in recipes:
        ids = [ingredient_to_id[item] for item in recipe if item in ingredient_to_id]

        # Need at least 3 ingredients for meaningful CBOW training
        if len(ids) >= 3:
            recipe_ids.append(ids)

    print(f"Recipes after vocabulary filtering: {len(recipe_ids):,}")
    return recipe_ids


# ----------------------------
# CBOW training examples
# ----------------------------

def make_cbow_examples(recipe_ids):
    """
    For each ingredient in a recipe:
    context = all other ingredients
    target = the selected ingredient

    Example:
    recipe = [flour, sugar, butter, egg]
    target = butter
    context = [flour, sugar, egg]
    """
    examples = []

    for recipe in recipe_ids:
        for index, target in enumerate(recipe):
            context = recipe[:index] + recipe[index + 1:]

            if len(context) > 0:
                examples.append((context, target))

    random.shuffle(examples)

    print(f"CBOW training examples: {len(examples):,}")
    return examples


def collate_batch(batch):
    """
    Pad variable-length context lists into a rectangular tensor.
    """
    contexts, targets = zip(*batch)

    max_len = max(len(context) for context in contexts)

    context_tensor = torch.zeros((len(contexts), max_len), dtype=torch.long)

    for row, context in enumerate(contexts):
        context_tensor[row, :len(context)] = torch.tensor(context, dtype=torch.long)

    target_tensor = torch.tensor(targets, dtype=torch.long)

    return context_tensor, target_tensor


# ----------------------------
# Model
# ----------------------------

class CBOWIngredientModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0
        )

        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        """
        context_ids shape:
        [batch_size, context_length]

        Output:
        logits for every ingredient in vocabulary.
        """
        embedded = self.embedding(context_ids)

        mask = (context_ids != 0).unsqueeze(-1)
        embedded = embedded * mask

        summed = embedded.sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1)

        context_vector = summed / counts

        logits = self.output_layer(context_vector)

        return logits


# ----------------------------
# Evaluation
# ----------------------------

def evaluate_top_k(model, examples, k=10, max_examples=2000):
    """
    Measures whether the hidden ingredient appears in the model's top-k guesses.
    This evaluates missing-ingredient prediction, not true substitution quality.
    """
    model.eval()

    sample = examples[:max_examples]
    hits = 0

    with torch.no_grad():
        for context, target in sample:
            context_tensor = torch.tensor([context], dtype=torch.long)

            logits = model(context_tensor)

            # Do not recommend ingredients already in the context
            for ingredient_id in context:
                logits[0, ingredient_id] = float("-inf")

            top_ids = torch.topk(logits, k=k).indices[0].tolist()

            if target in top_ids:
                hits += 1

    return hits / len(sample)


# ----------------------------
# Recommendation
# ----------------------------

def recommend_candidates(
    model,
    ingredient_to_id,
    id_to_ingredient,
    context_ingredients,
    removed_ingredient=None,
    top_k=10
):
    """
    Given a recipe context, rank possible ingredients that fit that context.

    If removed_ingredient is provided, it is excluded from the recommendations.
    That way the output can be interpreted as possible alternatives.
    """
    model.eval()

    cleaned_context = []

    for item in context_ingredients:
        cleaned = clean_ingredient(item)
        if cleaned is not None:
            cleaned_context.append(cleaned)

    if removed_ingredient is not None:
        removed_clean = clean_ingredient(removed_ingredient)
        cleaned_context = [x for x in cleaned_context if x != removed_clean]
    else:
        removed_clean = None

    context_ids = [
        ingredient_to_id[item]
        for item in cleaned_context
        if item in ingredient_to_id
    ]

    if len(context_ids) == 0:
        print("No usable ingredients from the context are in the vocabulary.")
        return []

    context_tensor = torch.tensor([context_ids], dtype=torch.long)

    with torch.no_grad():
        logits = model(context_tensor)

        # Exclude ingredients already in the recipe context
        for ingredient_id in context_ids:
            logits[0, ingredient_id] = float("-inf")

        # Exclude the removed ingredient itself
        if removed_clean in ingredient_to_id:
            logits[0, ingredient_to_id[removed_clean]] = float("-inf")

        probabilities = torch.softmax(logits, dim=1)

        top = torch.topk(probabilities, k=top_k)

    results = []

    for ingredient_id, score in zip(top.indices[0].tolist(), top.values[0].tolist()):
        results.append((id_to_ingredient[ingredient_id], score))

    return results


# ----------------------------
# Main script
# ----------------------------

def main():
    if not CSV_PATH.exists():
        print(f"Could not find {CSV_PATH}")
        raise SystemExit

    recipes = load_recipes(CSV_PATH, MAX_RECIPES)

    ingredient_to_id, id_to_ingredient = build_vocabulary(
        recipes,
        MIN_INGREDIENT_FREQ
    )

    recipe_ids = recipes_to_id_lists(recipes, ingredient_to_id)

    examples = make_cbow_examples(recipe_ids)

    split_index = int(len(examples) * (1 - TEST_FRACTION))

    train_examples = examples[:split_index]
    test_examples = examples[split_index:]

    print(f"Training examples: {len(train_examples):,}")
    print(f"Testing examples: {len(test_examples):,}")

    train_loader = DataLoader(
        train_examples,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch
    )

    model = CBOWIngredientModel(
        vocab_size=len(ingredient_to_id),
        embedding_dim=EMBEDDING_DIM
    )

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining model...")

    for epoch in range(EPOCHS):
        model.train()

        total_loss = 0

        for context_batch, target_batch in train_loader:
            optimizer.zero_grad()

            logits = model(context_batch)

            loss = loss_function(logits, target_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        top_10_accuracy = evaluate_top_k(
            model,
            test_examples,
            k=10,
            max_examples=1000
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss = {average_loss:.4f} | "
            f"test top-10 accuracy = {top_10_accuracy:.3f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "ingredient_to_id": ingredient_to_id,
            "id_to_ingredient": id_to_ingredient,
            "embedding_dim": EMBEDDING_DIM,
        },
        MODEL_OUTPUT_PATH
    )

    print(f"\nSaved model to {MODEL_OUTPUT_PATH}")

    # Demo example
    print("\nExample recommendation:")

    demo_recipe = [
        "flour",
        "sugar",
        "egg",
        "vanilla",
        "butter"
    ]

    removed = "butter"

    print("Recipe context:", demo_recipe)
    print("Removed ingredient:", removed)

    recommendations = recommend_candidates(
        model=model,
        ingredient_to_id=ingredient_to_id,
        id_to_ingredient=id_to_ingredient,
        context_ingredients=demo_recipe,
        removed_ingredient=removed,
        top_k=10
    )

    print("\nCandidate substitutions / context-based alternatives:")

    for rank, (ingredient, score) in enumerate(recommendations, start=1):
        print(f"{rank}. {ingredient}   score={score:.4f}")


if __name__ == "__main__":
    main()