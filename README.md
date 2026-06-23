# Summary
Our model is a CBOW-style neural network adapted from Word2Vec for ingredient prediction. Each recipe is treated as a collection of ingredient tokens, and during training one ingredient is removed from the recipe and treated as the target ingredient. The remaining ingredients form the context input. Each context ingredient is converted into a learned embedding vector of dimension $d$, and these vectors are averaged to create one context vector representing the recipe without the removed ingredient. This context vector is passed through a linear output layer, producing one score for every ingredient in the vocabulary. A SoftMax function converts these scores into probabilities, so the model ranks which ingredients are most likely to fit the given recipe context. At prediction time, the original removed ingredient and any ingredients already in the recipe are excluded from the final list, and the highest-probability remaining ingredients are interpreted as candidate substitutions or context-based alternatives.
# Sources
## SoftMax Function
[Softmax Function Explained in Depth with 3D Visuals](https://www.youtube.com/watch?v=ytbYRIN0N4g)
[Softmax Wikipedia Entry](https://en.wikipedia.org/wiki/Softmax_function)
## Word Embeddings
[Explanation of Word Embeddings](https://cbail.github.io/textasdata/word2vec/rmarkdown/word2vec.html)

# Code Breakdown
## Trainable Parameters

| Symbol               | Meaning                                                                 |
| -------------------- | ----------------------------------------------------------------------- |
| $E$                  | embedding matrix; stores the learned vector for each ingredient         |
| $d$                  | embedding dimension, or number of coordinates in each ingredient vector |
| $W$                  | output weight matrix that maps the context vector to ingredient scores  |
| $b$                  | output bias vector                                                      |
| $\theta = {E, W, b}$ | all trainable model parameters                                          |
## Main Model Variables
| Symbol | Meaning                                                        |
| ------ | -------------------------------------------------------------- |
| $V$    | ingredient vocabulary                                          |
| $r$    | one recipe                                                     |
| $i$    | selected/removed ingredient                                    |
| $C_i$  | context ingredients, meaning the recipe ingredients except $i$ |
| $v_j$  | embedding vector for context ingredient $j$                    |
| $h$    | averaged context vector                                        |
| $z_j$  | raw score/logit for candidate ingredient $j$                   |
| $p_j$  | SoftMax probability assigned to candidate ingredient $j$       |
| $k$    | number of recommendations returned                             |
## Main Hyperparameters
| Hyperparameter        | Meaning                                                                |
| --------------------- | ---------------------------------------------------------------------- |
| `MAX_RECIPES`         | number of recipes used for training                                    |
| `MIN_INGREDIENT_FREQ` | minimum frequency required for an ingredient to stay in the vocabulary |
| `EMBEDDING_DIM`       | dimension $d$ of the ingredient vectors                                |
| `EPOCHS`              | number of passes through the training data                             |
| `BATCH_SIZE`          | number of training examples processed at once                          |
| `LEARNING_RATE`       | step size used by the optimizer                                        |
| $k$                   | number of output recommendations shown                                 |
