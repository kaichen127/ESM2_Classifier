# 🧬 Post-Translational Modification (PTM) — ESM2 Classifier

This project uses Facebook’s [ESM2](https://github.com/facebookresearch/esm) protein language model as a backbone to perform **classification of PTM (Post-Translational Modification) sites** in protein sequences.

## 🚀 Features

- ⚙️ Built on top of **Facebook’s ESM2** transformer model.
- 🧩 Supports **multi-PTM classification** through an optional embedding table and Transformer decoder head.
- 🛠 Adjustable configurations via configs/config.yaml including:
  - ESM model size
  - Multi-PTM embedding
  - Custom decoder layers
  - Training hyperparameters
- 📁 Includes a **sample CSV dataset** from the **UniProt Consortium** for quick testing and exploration.

## 📂 Dataset

A small example CSV dataset is provided from the [UniProt](https://www.uniprot.org/) database, including:
- Protein sequences
- Site labels

## 📝 Usage

Modify configs.yaml, dataset.py, and model.py as needed and run train.py to begin training. Included in training are multiple toggleable flags that allow for loading checkpoints, visualization, and evaluating on the test set. 

## 📚 Reference

- [UniProt Consortium](https://www.uniprot.org/)
- [ESM2 by Meta AI](https://github.com/facebookresearch/esm)
