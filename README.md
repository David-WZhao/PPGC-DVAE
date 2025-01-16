# PPGC-DVAE

This is the codes of the PPGC-DVAE in our manuscript "A Conditional Denoising VAE-based Framework for Antimicrobial Peptides Generation with Reserving Desirable Properties".

### requirement

+ python == 3.8.15
+ torch == 1.12.1
+ rdkit-pypi == 2022.9.5
+ tqdm == 4.64.1
+ biopython == 1.83

### Dataset

The dataset used in this paper is located in the `data/` directory. The `train_last_new.csv` is the training set, `valid_last_new.csv` is the validation set, and `test_last_new.csv` is the test set. These three files have been processed and are ready for use; you can download them for training directly.

### Training

The entire training process takes approximately one and a half hours on a 3090 GPU with 32GB of VRAM.

```
python train.py
```

Make sure the data file is ready before running!

If you want to replace it with your own dataset, please prepare it as follows:

+ Prepare the corresponding antimicrobial peptide (AMP) sequences and save them in CSV format. You can refer to `data/` for guidance.

+ Open `predict1.py` in the `utils/` directory and modify the corresponding line of code. The same applies to the validation and test sets.

  ```
  fasta_file1 = "data/train.fasta"  # Replace it with the path to your FASTA file.
  output_file1 = "data/train.csv"  # Output the path to the CSV file.
  ```

Of course, if you plan to use the model directly without training, the model's weight file `best_model1.pth` is also provided here.

### Generate

```
python newgenerate.py
```

If you need to modify the target condition values, you can do so in this file. If you need to normalize the target input values, you can use `utils/csvguiyihua.py`.

