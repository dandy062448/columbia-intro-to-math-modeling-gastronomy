"""
Simple ingredient co-occurrence substitution recommender.

Given a recipe (a list of ingredients) and one ingredient the user wants to
substitute, this script recommends the top 10 ingredients that most often
appear alongside the *other* ingredients in that recipe, based on how often
ingredient pairs co-occur across the RecipeNLG dataset.

No machine learning / embeddings here -- just counting how often ingredient
pairs show up together in the same recipe (a co-occurrence matrix), which is
a common, simple baseline for "people who used X also used Y" style
recommendations.

Usage
-----
Run interactively:
    python cooccurrence_substitution_model.py

Or pass the recipe and the ingredient to replace directly:
    python cooccurrence_substitution_model.py \
        --recipe "flour, sugar, butter, eggs, milk, vanilla" \
        --substitute butter
"""

from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import argparse
import ast

import pandas as pd


# ----------------------------
# Settings
# ----------------------------

CSV_PATH = Path("./RecipeNLG_dataset.csv")

MAX_RECIPES = 100_000       # cap so this stays fast on the full 2.2M-row file
MIN_INGREDIENT_FREQ = 10    # ignore ingredients that barely ever appear
TOP_K = 10


# ----------------------------
# Data loading / cleaning
# (mirrors the approach used in model2.py so results are comparable)
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
    Basic ingredient cleaning. Kept intentionally simple.
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
    Load a limited number of recipes from RecipeNLG using only the NER
    column. Each recipe becomes a list of cleaned, deduplicated ingredient
    tokens.
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

        # Keep only recipes with enough context to be useful
        if len(cleaned) >= 2:
            recipes.append(cleaned)

    print(f"Usable recipes: {len(recipes):,}")
    return recipes


def filter_by_frequency(recipes, min_freq):
    """
    Drop rare ingredients (likely typos / one-off phrasing) so the
    co-occurrence counts aren't dominated by noise.
    """
    counts = Counter()
    for recipe in recipes:
        counts.update(recipe)

    keep = {ingredient for ingredient, count in counts.items() if count >= min_freq}

    filtered = []
    for recipe in recipes:
        kept = [item for item in recipe if item in keep]
        if len(kept) >= 2:
            filtered.append(kept)

    print(f"Unique ingredients before filtering: {len(counts):,}")
    print(f"Unique ingredients kept (freq >= {min_freq}): {len(keep):,}")
    print(f"Recipes remaining after filtering: {len(filtered):,}")

    return filtered, counts


# ----------------------------
# Co-occurrence model
# ----------------------------

def build_cooccurrence(recipes):
    """
    Count how often every pair of ingredients appears together in the same
    recipe, and how often each ingredient appears overall.

    Returns
    -------
    cooccurrence : dict[str, Counter]
        cooccurrence[a][b] = number of recipes containing both a and b
    ingredient_counts : Counter
        ingredient_counts[a] = number of recipes containing a
    """
    cooccurrence = defaultdict(Counter)
    ingredient_counts = Counter()

    for recipe in recipes:
        ingredient_counts.update(recipe)

        for a, b in combinations(sorted(set(recipe)), 2):
            cooccurrence[a][b] += 1
            cooccurrence[b][a] += 1

    print(f"Built co-occurrence table for {len(cooccurrence):,} ingredients.")
    return cooccurrence, ingredient_counts


# ----------------------------
# Recommendation
# ----------------------------

def clean_recipe_ingredients(recipe_ingredients):
    """
    Clean and deduplicate a raw list of ingredient strings the same way
    the training data was cleaned.
    """
    cleaned_recipe = []
    for item in recipe_ingredients:
        cleaned = clean_ingredient(item)
        if cleaned is not None:
            cleaned_recipe.append(cleaned)
    return list(dict.fromkeys(cleaned_recipe))


def ingredient_in_recipe(cleaned_recipe, ingredient_to_substitute):
    """
    Check whether `ingredient_to_substitute` (after the same cleaning used
    for the recipe) is actually present in the cleaned recipe.
    """
    target = clean_ingredient(ingredient_to_substitute)
    return target is not None and target in cleaned_recipe


def _prepare_context(cleaned_recipe, ingredient_to_substitute, cooccurrence):
    """
    Shared setup for both scoring methods: figure out the context
    (recipe minus the substituted ingredient), which context ingredients
    are actually known to the model, and which candidates should be
    excluded from the results.
    """
    target = clean_ingredient(ingredient_to_substitute)
    context = [item for item in cleaned_recipe if item != target]
    known_context = [item for item in context if item in cooccurrence]
    exclude = set(context) | {target}
    return context, known_context, exclude


def recommend_substitutes(
    cooccurrence,
    ingredient_counts,
    recipe_ingredients,
    ingredient_to_substitute,
    top_k=TOP_K,
):
    """
    Recommend replacement ingredients for `ingredient_to_substitute`,
    ranked by raw co-occurrence count with the *rest* of the recipe
    (i.e. the recipe with that ingredient removed).

    Returns
    -------
    list[tuple[str, float]]
        (ingredient, score) pairs, sorted by descending score.
    """
    cleaned_recipe = clean_recipe_ingredients(recipe_ingredients)

    _, known_context, exclude = _prepare_context(
        cleaned_recipe, ingredient_to_substitute, cooccurrence
    )

    if not known_context:
        print("None of the remaining recipe ingredients were found in the dataset.")
        return []

    scores = Counter()
    for context_ingredient in known_context:
        for candidate, count in cooccurrence[context_ingredient].items():
            if candidate in exclude:
                continue
            scores[candidate] += count

    if not scores:
        print("No co-occurring candidates found for this recipe.")
        return []

    return scores.most_common(top_k)


def recommend_substitutes_jaccard(
    cooccurrence,
    ingredient_counts,
    recipe_ingredients,
    ingredient_to_substitute,
    top_k=TOP_K,
):
    """
    Recommend replacement ingredients for `ingredient_to_substitute`,
    ranked by Jaccard similarity with the *rest* of the recipe.

    For a pair of ingredients (a, b):

        jaccard(a, b) = cooccurrence(a, b) / (count(a) + count(b) - cooccurrence(a, b))

    This normalizes raw co-occurrence by how common each ingredient is
    overall, so very common ingredients (salt, sugar, flour, ...) don't
    automatically dominate the recommendations just because they appear
    in almost every recipe.

    Returns
    -------
    list[tuple[str, float]]
        (ingredient, score) pairs, sorted by descending score.
    """
    cleaned_recipe = clean_recipe_ingredients(recipe_ingredients)

    _, known_context, exclude = _prepare_context(
        cleaned_recipe, ingredient_to_substitute, cooccurrence
    )

    if not known_context:
        print("None of the remaining recipe ingredients were found in the dataset.")
        return []

    scores = Counter()
    for context_ingredient in known_context:
        context_count = ingredient_counts[context_ingredient]

        for candidate, count in cooccurrence[context_ingredient].items():
            if candidate in exclude:
                continue

            union = context_count + ingredient_counts[candidate] - count
            if union <= 0:
                continue

            scores[candidate] += count / union

    if not scores:
        print("No co-occurring candidates found for this recipe.")
        return []

    return scores.most_common(top_k)


# ----------------------------
# Main script
# ----------------------------

def build_model(csv_path=CSV_PATH, max_recipes=MAX_RECIPES, min_freq=MIN_INGREDIENT_FREQ):
    """
    Convenience helper: load the CSV, clean it, filter it, and build the
    co-occurrence table in one call. Returns (cooccurrence, ingredient_counts).
    """
    if not Path(csv_path).exists():
        print(f"Could not find {csv_path}")
        raise SystemExit(1)

    recipes = load_recipes(csv_path, max_recipes)
    recipes, _ = filter_by_frequency(recipes, min_freq)
    cooccurrence, ingredient_counts = build_cooccurrence(recipes)

    return cooccurrence, ingredient_counts


def parse_recipe_string(recipe_string):
    """Turn a comma-separated string of ingredients into a list."""
    return [item.strip() for item in recipe_string.split(",") if item.strip()]


def get_recipe_from_user(initial_recipe_string=None):
    """
    Prompt the user for a recipe (or use a value already supplied via
    --recipe on the first loop iteration) and return both the raw
    ingredient list and the cleaned version used for validation/lookup.
    Re-prompts if the recipe is empty or nothing survives cleaning.
    """
    recipe_string = initial_recipe_string

    while True:
        if recipe_string is None:
            recipe_string = input(
                "\nEnter your recipe's ingredients, separated by commas:\n> "
            )

        recipe_ingredients = parse_recipe_string(recipe_string)
        cleaned_recipe = clean_recipe_ingredients(recipe_ingredients)

        if cleaned_recipe:
            return recipe_ingredients, cleaned_recipe

        print("That recipe didn't contain any usable ingredients. Please try again.")
        recipe_string = None


def get_substitute_from_user(cleaned_recipe, initial_substitute=None):
    """
    Prompt the user for the ingredient to substitute and validate that it
    is actually present in the (cleaned) recipe. Re-prompts until a valid
    ingredient is given.
    """
    substitute_target = initial_substitute

    while True:
        if substitute_target is None:
            substitute_target = input(
                "\nWhich ingredient do you want to substitute?\n> "
            ).strip()

        if ingredient_in_recipe(cleaned_recipe, substitute_target):
            return substitute_target

        print(
            f"Error: '{substitute_target}' was not found in the recipe you provided "
            f"({', '.join(cleaned_recipe)}). Please enter an ingredient that is in the recipe."
        )
        substitute_target = None


def print_recommendations(title, recommendations, substitute_target):
    if recommendations:
        print(f"\n{title} for '{substitute_target}':")
        for rank, (ingredient, score) in enumerate(recommendations, start=1):
            print(f"{rank}. {ingredient}   (score={score:.4f})" if isinstance(score, float)
                  else f"{rank}. {ingredient}   (score={score})")
    else:
        print(f"\n{title}: no recommendations could be generated for this input.")


def prompt_yes_no(question):
    answer = input(f"\n{question} (y/n): ").strip().lower()
    return answer.startswith("y")


def main():
    parser = argparse.ArgumentParser(
        description="Recommend ingredient substitutions using a co-occurrence model."
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Comma-separated list of ingredients, e.g. 'flour, sugar, butter, eggs'",
    )
    parser.add_argument(
        "--substitute",
        type=str,
        default=None,
        help="The ingredient in the recipe you want to replace, e.g. 'butter'",
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=str(CSV_PATH),
        help="Path to RecipeNLG_dataset.csv",
    )
    parser.add_argument(
        "--max-recipes",
        type=int,
        default=MAX_RECIPES,
        help="Max number of recipes to read from the CSV",
    )
    args = parser.parse_args()

    cooccurrence, ingredient_counts = build_model(
        csv_path=args.csv_path,
        max_recipes=args.max_recipes,
    )

    # Only the first loop iteration can be pre-filled from CLI args;
    # every later round always prompts interactively.
    next_recipe_string = args.recipe
    next_substitute = args.substitute

    while True:
        recipe_ingredients, cleaned_recipe = get_recipe_from_user(next_recipe_string)
        substitute_target = get_substitute_from_user(cleaned_recipe, next_substitute)

        # CLI-supplied values are only used once
        next_recipe_string = None
        next_substitute = None

        print(f"\nRecipe: {', '.join(recipe_ingredients)}")
        print(f"Ingredient to substitute: {substitute_target}")

        cooccurrence_recs = recommend_substitutes(
            cooccurrence,
            ingredient_counts,
            recipe_ingredients,
            substitute_target,
            top_k=TOP_K,
        )
        print_recommendations(
            f"Top {TOP_K} substitution candidates (raw co-occurrence)",
            cooccurrence_recs,
            substitute_target,
        )

        jaccard_recs = recommend_substitutes_jaccard(
            cooccurrence,
            ingredient_counts,
            recipe_ingredients,
            substitute_target,
            top_k=TOP_K,
        )
        print_recommendations(
            f"Top {TOP_K} substitution candidates (Jaccard similarity)",
            jaccard_recs,
            substitute_target,
        )

        if not prompt_yes_no("Would you like to try another recipe and ingredient?"):
            print("\nDone. Happy cooking!")
            break


if __name__ == "__main__":
    main()
