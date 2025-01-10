from rdkit import Chem
from rdkit.Chem import AllChem

# sequence = "KWCFRVCYRGICYRRCR" #Tachyplesin
sequence = "EGKKTFFIQGGF"
mol = Chem.MolFromFASTA(sequence)
smiles = Chem.MolToSmiles(mol)
print(smiles)