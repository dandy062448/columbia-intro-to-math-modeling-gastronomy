## Comparison Table
Add a column for each simple type of test. If we wind up using a bunch, move the interpretation into a section for our report and the verbal part of our presentation. Other methods to include could be Jaccard or lift top-10 recs. 

| Input recipe context       | Ingredient removed | CBOW recommendations             | Co-occurrence recommendations | Interpretation                                                                                                  |
| -------------------------- | ------------------ | -------------------------------- | ----------------------------- | --------------------------------------------------------------------------------------------------------------- |
| flour, sugar, egg, vanilla | butter             | margarine, shortening, Crisco... | sugar, flour, salt, milk...   | CBOW gives more functional substitutions<br><br>Co-occurrence gives common baking companions, not substitutions |
## Tests to Run

Test a single recipe, cycling through each ingredient. Test ingredients depending on "categories" like fats or herbs.

| Removed ingredient | Recipe context                     | Why it is useful                                                                                                |
| ------------------ | ---------------------------------- | --------------------------------------------------------------------------------------------------------------- |
| butter             | flour, sugar, egg, vanilla         | Tests baking fat substitutions                                                                                  |
| milk               | flour, sugar, egg, vanilla, butter | Test the rest of each ingredient to see if there are any noticeable differences in the "quality" of suggestions |
| chicken            | garlic, onion, rice, broth         | Tests protein substitutions                                                                                     |
| cilantro           | lime, onion, tomato, jalapeno      | Tests herb and spice substitutions                                                                              |
| soy sauce          | ginger, garlic, sesame oil, rice   | Tests sauce and maybe umami substitutions                                                                       |
## Top-k 
You have added the top-1 and top-5 to our top-10. I think our CSV could be clipped down to show some of the results we test above to highlight the success of our model if given the ability to make more recommendations.

## Sensitivity Analysis
I have a few examples of hyperparameters that I changed that I included screenshots of in the last assignment. We could start there, comparing the loss or subjective "accuracy" of recommendations.
- training set size vs top-10 "accuracy"
- embedding dimension $d$ vs top-10 "accuracy"
- epochs vs loss
- epochs vs top-10 "accuracy"

## Limitations
I think I can speak to this more, but I already wrote a paragraph or two about it in the last assignment. I compared the difference between a contextual suggestion and a practical real-life suggestions. Let me know if you have any other ideas. I can touch on RecipeNLG not having flavor tags or other categories for each item in `NER`. 

## Visualizations
- plot training loss over epochs
- plot top-k "accuracy" over epochs
- bar graphs comparing top-k accuracy against (10,000 , 100,000 , 500,000 recipes) ($d$ = 25, 50, 100) (epochs = 3, 5, 10)
- tables mentioned above
