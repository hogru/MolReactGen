These are sample reaction templates / SMARTS. The content in the CSV and PDF is the same.

The table has the following columns:

- **feasible_reaction_smarts:** The generated string / reaction template / SMARTS, already filtered by checking if it can be parsed by `rdkit`
- **not_trained_on_but_known:** TRUE if the reaction template is in the validation or test set (but not in the training set)
- **known_from_valid:** TRUE if the reaction template is in the validation set
- **known_from_test:** TRUE if the reaction template is in the test set; note that USPTO-50K is non-disjunct w.r.t. reaction templates
- **num_products_works_with:** Number of products found in USPTO-50K that the reaction template can be applied to, i.e. result in one or more reactant(s) with `rdkit`. There are potentially more products since products are only searched for with a simple heuristic and not with a comprehensive search. If this number is ≥ 1 the reaction template is considered to be _feasible_
- **example_works_with_reaction_id:** one (of potentially many) USPTO-50K ID(s) the reaction template can be applied to

The following metrics are used (RTS = Reaction Templates):

- **Validity:** $\frac{RTS_{valid}}{RTS_{generated}}$ with only using the first 10,000 reaction templates; a reaction template is considered valid, if all reactants and products can be parsed by `rdkit`
- **Uniqueness:** $\frac{RTS_{unique}}{RTS_{valid}}$
- **Feasibility:** $\frac{RTS_{feasible}}{RTS_{unique}}$
- **Known:** Number of reaction templates being either in validation or test set, validation set only, test set only

These reaction templates result in the following metric values:

| Metric                                   | Value |
| ---------------------------------------- | ----- |
| Validity                                 | 0.748 |
| Uniqueness                               | 0.838 |
| Feasibility                              | 0.105 |
| Known from either validation or test set | 708   |
| Known from validation set                | 496   |
| Known from test set                      | 480   |
