These are sample molecules / SMILES. The content in the CSV and PDF is the same.

The table has the following columns:

- **smiles:** The generated string / molecule / SMILES
- **canonical_smiles:** The canonicalized form of the smiles column. For non-valid smiles this is empty
- **valid:** TRUE if the molecule can be parsed with `rdkit`
- **unique:** TRUE if the molecule was generated only once
- **novel:** TRUE if the molecule is not in the training set

The following metrics are used (Mols = Molecules):

- **Validity:** $\frac{Mols_{valid}}{Mols_{generated}}$ with only using the first 10,000 molecules
- **Uniqueness:** $\frac{Mols_{unique}}{Mols_{valid}}$
- **Novelty:** $\frac{Mols_{novel}}{Mols_{unique}}$
- **Fréchet ChemNet Distance (FCD):** a similarity measure between two sets of molecules, in this case the GuacaMol training set and the generated molecules. For details please refer to this [paper](https://pubmed.ncbi.nlm.nih.gov/30118593/)
- **FCD GuacaMol style:** $FCD_{GuacaMol} = e^{-0.2 FCD}$

These molecules result in the following metric values:

| Metric       | _Valid_ molecules | _Novel_ molecules |
| ------------ | ----------------- | ----------------- |
| Validity     | 0.976             |                   |
| Uniqueness   | 0.999             |                   |
| Novelty      | 0.940             |                   |
| FCD          | 0.216             | 0.222             |
| FCD_Guacamol | 0.958             | 0.957             |

The FCD is slightly worse if we only use _novel_ molecules. This can be expected since _valid_ molecules also contain _non-novel_ molecules, i.e. they are contained in the training set. And molecules from the training set have (by definition) a (close to) zero FCD.
