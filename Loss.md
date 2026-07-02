# Loss
In your code, **loss** is the number the model is trying to minimize during training:

```python
loss_function = nn.CrossEntropyLoss()
```

Your model predicts one missing/target ingredient from the surrounding context ingredients, and `CrossEntropyLoss` measures how wrong that prediction was. 

Mathematically, for one training example, your model does this:

[
\text{context ingredients} \rightarrow \text{context vector} \rightarrow \text{score for every ingredient}
]

The output layer produces one score for every ingredient in the vocabulary:

[
z_1, z_2, z_3, \dots, z_V
]

where (V) is the vocabulary size.

Those raw scores are called **logits**. The model then turns them into probabilities using softmax:

[
p_j = \frac{e^{z_j}}{\sum_{k=1}^{V} e^{z_k}}
]

So (p_j) is the model’s predicted probability that ingredient (j) is the missing target ingredient.

If the true missing ingredient is `"butter"`, then the loss is:

[
\text{loss} = -\log(p_{\text{butter}})
]

So if the model gives `"butter"` a high probability, the loss is low. If it gives `"butter"` a low probability, the loss is high.

For example:

| Probability assigned to correct ingredient |  Loss |
| -----------------------------------------: | ----: |
|                                       1.00 |     0 |
|                                       0.50 | 0.693 |
|                                       0.10 | 2.303 |
|                                       0.05 | 2.996 |
|                                       0.01 | 4.605 |
|                                      0.001 | 6.908 |

So a loss of **0** would mean the model is perfectly confident in the correct answer. That almost never happens in this kind of problem.

Your reported loss is an average over many training examples. So when you got:

[
\text{Final Loss} = 3.7107
]

that means the model’s average negative log-probability for the correct ingredient was about 3.7107.

One way to interpret that is:

[
e^{-3.7107} \approx 0.0245
]

So the model assigned the correct ingredient a geometric average probability of about **2.45%**. That may sound low, but remember that the model is choosing from thousands of possible ingredients. In your 100,000-recipe run, the vocabulary size was 2,011 ingredients. A totally uniform random guess would assign each ingredient probability:

[
\frac{1}{2011} \approx 0.0005
]

That corresponds to a random baseline loss of:

[
-\log(1/2011) = \log(2011) \approx 7.61
]

So a loss around **3.71** is much better than random guessing.

The reason loss supports your model is that your training task is based on **co-occurrence**. In your `make_cbow_examples()` function, each recipe is turned into context-target pairs. For example:

```text
recipe = flour, sugar, butter, eggs, milk, vanilla
target = butter
context = flour, sugar, eggs, milk, vanilla
```

The model is trained to predict `"butter"` from the ingredients that appear with it. If lowering the loss means the model is getting better at predicting missing ingredients from their recipe contexts, then the model is learning patterns of ingredient co-occurrence.

That is why loss is relevant to your research question. It does **not** prove that the model understands true culinary substitutions. It shows that the model has learned which ingredients tend to fit similar recipe contexts. Then, when you remove `"butter"` and ask for alternatives, the model recommends ingredients that receive high predicted probability in that same context.

So the logic is:

[
\text{lower loss} \Rightarrow \text{better prediction of missing ingredients from context}
]

and since the context is made of co-occurring recipe ingredients:

[
\text{better prediction from context} \Rightarrow \text{better learned co-occurrence structure}
]

For your report, you could write:

> The loss function measures how well the model predicts a held-out ingredient from the remaining ingredients in the recipe. We use cross-entropy loss, which penalizes the model when it assigns low probability to the correct target ingredient. Because the training examples are constructed from recipe co-occurrence, decreasing loss indicates that the model is learning which ingredients tend to appear in similar recipe contexts. Therefore, loss provides evidence that the model is learning useful co-occurrence patterns, although it does not by itself prove that the recommendations are true substitutions.

For good and bad loss values, there is no universal cutoff. It depends heavily on vocabulary size. A loss of 4 might be good when there are 2,000 possible ingredients, but worse in a much smaller vocabulary.

A useful rule for your project:

| Loss pattern                                              | Interpretation                                                |
| --------------------------------------------------------- | ------------------------------------------------------------- |
| Loss decreases over epochs                                | The model is learning from the data.                          |
| Loss stays almost the same                                | The model is not improving much.                              |
| Loss is close to (\log(V))                                | The model is close to random guessing.                        |
| Loss is much lower than (\log(V))                         | The model is learning meaningful prediction patterns.         |
| Training loss decreases but test accuracy stops improving | The model may be overfitting or reaching diminishing returns. |

For your runs, the loss values are behaving well. For example, in the 100,000-recipe, 5-epoch run:

```text
Epoch 1 loss = 4.8969
Epoch 5 loss = 3.7107
```

That is a meaningful decrease. At the same time, Top-10 accuracy increased:

```text
Top-10: 0.536 → 0.628
```

That combination is important. Loss alone is not enough, but loss decreasing while Top-1, Top-5, and Top-10 accuracy increase is good evidence that the model is learning a useful relationship between context ingredients and target ingredients.

For your paper, I would not say “low loss proves the model finds good substitutions.” I would say:

> Lower loss shows that the model became better at predicting ingredients from recipe context. Since recipe context is based on ingredient co-occurrence, this supports the claim that the model learned co-occurrence patterns. The recommendation quality still needs to be evaluated separately using Top-1, Top-5, Top-10 accuracy and qualitative examples.
