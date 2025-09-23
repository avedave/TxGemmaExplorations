def BindingDB_ic50_prompt(drug_smiles, target_amino_acid_sequence):
    task_name = "BindingDB_ic50"
    prompt_template = "Instructions: Answer the following question about drug target interactions.\nContext: Drug-target binding is the physical interaction between a drug and a specific biological molecule, such as a protein or enzyme. This interaction is essential for the drug to exert its pharmacological effect. The strength of the drug-target binding is determined by the binding affinity, which is a measure of how tightly the drug binds to the target. IC50 is the concentration of a drug that inhibits a biological activity by 50%. It is a measure of the drug's potency, but it is not a direct measure of binding affinity. \nQuestion: Given the target amino acid sequence and compound SMILES string, predict their normalized binding affinity Kd from 000 to 1000, where 000 is minimum IC50 and 1000 is maximum IC50.\nDrug SMILES: {Drug SMILES}\nTarget amino acid sequence: {Target amino acid sequence}\nAnswer:"
    values = {
        "Drug SMILES": drug_smiles,
        "Target amino acid sequence": target_amino_acid_sequence,
    }
    return prompt_template.format(**values)
