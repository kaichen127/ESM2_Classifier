import torch
from torch.utils.data import Dataset, DataLoader
import ast
import pandas as pd
from transformers import AutoTokenizer
import os

def build_condition2idx(ptm_types):
    return {ptm: idx for idx, ptm in enumerate(sorted(set(ptm_types)))}

def idx2condition(ptm_types):
    return {idx: ptm for idx, ptm in enumerate(sorted(set(ptm_types)))}

class PTMDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, subset_size=None):
        """
        Args:
            file_path (str): Path to the CSV file.
            tokenizer: Tokenizer from Hugging Face (e.g., ESM2 tokenizer).
            max_length (int): Maximum length for tokenized sequences.
        """

        self.data = pd.read_csv(file_path)

        if subset_size is not None:
            self.data = self.data.sample(n=subset_size, random_state=42).reset_index(drop=True)


        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sequence = row['Sequence']
        # substract 1 from position to go to 0-index from 1-index
        positions = [pos - 1 for pos in ast.literal_eval(row['position'])]
        amino_acids = ast.literal_eval(row['amino acid'])

        inputs = self.tokenizer(sequence, return_tensors="pt", padding="max_length", truncation=True,
                                max_length=self.max_length, add_special_tokens=False)

        labels = torch.zeros(self.max_length, dtype=torch.long)
        for pos in positions:
            if pos < self.max_length:
                labels[pos] = 1

        # print("Sequence: ", sequence)
        # print("Positions: ", positions)
        # print("Amino Acids: ", amino_acids)
        # print("Labels: ", labels)

        mask = torch.zeros(self.max_length, dtype=torch.bool)
        for pos, aa in zip(positions, amino_acids):
            if pos < self.max_length and sequence[pos] == aa:
                mask[pos] = True

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "mask": mask
        }

class MultiPTMDataset(Dataset):
    def __init__(self, ptm_list, data_root, tokenizer, condition2idx, split="train", max_length=512, subset_size=None):
        """
        Args:
            ptm_list (list): List of PTM types like ["Phosphorylation", "Methylation", ...]
            data_root (str): Root directory containing split folders (train/valid/test) each with PTM CSVs
            tokenizer: Hugging Face tokenizer.
            condition2idx (dict): Mapping from condition name to index.
            split (str): One of 'train', 'valid', or 'test'.
            max_length (int): Max sequence length.
        """
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.condition2idx = condition2idx

        for ptm_type in ptm_list:
            file_path = os.path.join(data_root, split, f"{ptm_type}.csv")  # e.g. ptm_data/train/Methylation.csv
            if not os.path.exists(file_path):
                print(f"[Warning] File not found: {file_path}")
                continue

            df = pd.read_csv(file_path)

            for _, row in df.iterrows():
                sequence = row['Sequence']
                try:
                    positions = [p - 1 for p in ast.literal_eval(str(row['position']))]
                    amino_acids = ast.literal_eval(str(row['amino acid']))
                except Exception as e:
                    print(f"Skipping bad row in {file_path}: {e}")
                    continue

                self.data.append({
                    "sequence": sequence,
                    "positions": positions,
                    "amino_acids": amino_acids,
                    "condition": ptm_type,
                    "condition_idx": condition2idx[ptm_type]
                })

        if subset_size is not None:
            import random
            self.data = random.sample(self.data, subset_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        inputs = self.tokenizer(entry["sequence"], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=self.max_length, add_special_tokens=False)

        labels = torch.zeros(self.max_length, dtype=torch.long)
        mask = torch.zeros(self.max_length, dtype=torch.bool)

        for pos, aa in zip(entry["positions"], entry["amino_acids"]):
            if pos < self.max_length and entry["sequence"][pos] == aa:
                labels[pos] = 1
                mask[pos] = True

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels,
            "mask": mask,
            "condition_idx": torch.tensor(entry["condition_idx"], dtype=torch.long)
        }


def prepare_dataloaders(configs, debug = False, debug_subset_size=None):
    """
    Prepares DataLoaders for training, validation, and testing based on configurations.
    Supports both singular and multi-PTM datasets.

    Args:
        configs: Configuration object containing file paths and DataLoader settings.

    Returns:
        dict: A dictionary containing DataLoaders for "train", "valid", and "test".
    """
    from transformers import AutoTokenizer

    model_name = configs.model.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataloaders = {}

    if configs.model.use_decoder_block:
        # MultiPTM dataset
        ptm_list = configs.ptm_types  # e.g., ["Phosphorylation", "Methylation"]
        condition2idx = build_condition2idx(ptm_list)
        data_root = configs.data_root

        for split in ["train", "valid", "test"]:
            dataset = MultiPTMDataset(
                ptm_list,
                data_root,
                tokenizer,
                condition2idx,
                split=split if split != "valid" else "test",
                max_length=configs.train_settings.max_sequence_length,
                subset_size=debug_subset_size if debug else None
            )
            dataloaders[split] = DataLoader(
                dataset,
                batch_size=configs.train_settings.batch_size,
                shuffle=(split == "train") and configs.train_settings.shuffle,
                num_workers=configs.train_settings.num_workers,
                pin_memory=True
            )
    else:
        # Singular PTM dataset
        if hasattr(configs, 'train_settings'):
            train_file = configs.train_settings.train_path
            batch_size = configs.train_settings.batch_size
            max_length = configs.train_settings.max_sequence_length
            shuffle = configs.train_settings.shuffle
            num_workers = configs.train_settings.num_workers

            train_dataset = PTMDataset(
                train_file,
                tokenizer,
                max_length=max_length,
                subset_size=debug_subset_size if debug else None
            )
            dataloaders["train"] = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True
            )

        # Prepare validation DataLoader
        if hasattr(configs, 'valid_settings'):
            valid_file = configs.valid_settings.valid_path
            batch_size = configs.valid_settings.batch_size
            max_length = configs.train_settings.max_sequence_length
            num_workers = configs.valid_settings.num_workers

            valid_dataset = PTMDataset(
                valid_file,
                tokenizer,
                max_length=max_length,
                subset_size=debug_subset_size if debug else None
            )
            dataloaders["valid"] = DataLoader(
                valid_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

        # Prepare test DataLoader
        if hasattr(configs, 'test_settings'):
            test_file = configs.test_settings.test_path
            batch_size = configs.test_settings.batch_size
            max_length = configs.train_settings.max_sequence_length
            num_workers = configs.test_settings.get('num_workers', 0)

            test_dataset = PTMDataset(
                test_file,
                tokenizer,
                max_length=max_length,
                subset_size=debug_subset_size if debug else None
            )
            dataloaders["test"] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True
            )

    return dataloaders

# just using this as a function to help determine what I should make the max_length
def analyze_protein_data(configs):
    import numpy as np
    from collections import Counter

    file_paths = [
        configs.train_settings.train_path,
        configs.valid_settings.valid_path,
        configs.test_settings.test_path
    ]

    all_lengths = []
    amino_acid_counts = Counter()
    total_phosphorylated = 0
    total_non_phosphorylated = 0

    print("Analyzing protein datasets...")
    for file_path in file_paths:
        if file_path:
            print(f"Reading data from {file_path}...")
            data = pd.read_csv(file_path)

            # Basic checks
            if "Sequence" not in data.columns or "position" not in data.columns:
                raise ValueError(f"The file {file_path} must contain 'Sequence' and 'position' columns.")

            lengths = data["Sequence"].apply(len).tolist()
            all_lengths.extend(lengths)

            for _, row in data.iterrows():
                sequence = row["Sequence"]
                # parse positions, subtract 1 for zero-based
                try:
                    positions = [p - 1 for p in ast.literal_eval(str(row["position"]))]
                except (ValueError, SyntaxError):
                    print(f"Skipping invalid position data: {row['position']}")
                    positions = []
                phosphorylated_count = len(positions)
                total_phosphorylated += phosphorylated_count
                total_non_phosphorylated += len(sequence) - phosphorylated_count

            # Count all amino acids (total distribution)
            for sequence in data["Sequence"]:
                amino_acid_counts.update(sequence)

            if 'amino acid' in data.columns:
                phospho_aa_set = set()
                for _, row in data.iterrows():
                    aa_list = eval(row["amino acid"])
                    phospho_aa_set.update(aa_list)

                phosphorylated_count_for_phospho_aa = 0
                unphosphorylated_count_for_phospho_aa = 0

                for _, row in data.iterrows():
                    sequence = row["Sequence"]
                    positions = [p - 1 for p in eval(row["position"])]

                    # loop through each residue in the sequence
                    for i, aa in enumerate(sequence):
                        # Only consider amino acids that appear in 'amino acid' column
                        if aa in phospho_aa_set:
                            if i in positions:
                                phosphorylated_count_for_phospho_aa += 1
                            else:
                                unphosphorylated_count_for_phospho_aa += 1

                print("Matching Amino Acids Statistics:")
                total_matched_aa = phosphorylated_count_for_phospho_aa + unphosphorylated_count_for_phospho_aa
                print(f"  - Amino acids in 'amino acid' col (set): {phospho_aa_set}")
                print(f"  - Total occurrences of these AAs in sequences: {total_matched_aa}")
                print(f"  - Phosphorylated (positions listed): {phosphorylated_count_for_phospho_aa}")
                print(f"  - Unphosphorylated (same AAs, other positions): {unphosphorylated_count_for_phospho_aa}")

                if total_matched_aa > 0:
                    prop_phospho = phosphorylated_count_for_phospho_aa / total_matched_aa
                    print(f"  - Proportion phosphorylated among these AAs: {prop_phospho:.4f}")
            else:
                print(f"No 'amino acid' column found in {file_path}, skipping matching-amino-acid stats.")

            print(f"Statistics for {file_path}:")
            print(f"  Total sequences: {len(lengths)}")
            print(f"  Median length: {np.median(lengths):.2f}")
            print(f"  Minimum length: {min(lengths)}")
            print(f"  Maximum length: {max(lengths)}")
            print()

    if all_lengths:
        all_lengths = np.array(all_lengths)
        all_lengths.sort()

        median_length = np.median(all_lengths)

        length_percentile = np.percentile(all_lengths, 95)

        total_residues = total_phosphorylated + total_non_phosphorylated
        perc_phosphorylated = (total_phosphorylated / total_residues) * 100
        perc_non_phosphorylated = (total_non_phosphorylated / total_residues) * 100

        weight_phosphorylated = total_residues / total_phosphorylated
        weight_non_phosphorylated = total_residues / total_non_phosphorylated

        print("Aggregate Statistics Across All Datasets:")
        print(f"  Total sequences: {len(all_lengths)}")
        print(f"  Median length: {median_length:.2f}")
        print(f"  Length capturing 95% of sequences: {length_percentile:.2f}")
        print(f"  Minimum length: {min(all_lengths)}")
        print(f"  Maximum length: {max(all_lengths)}")
        print(f"  Total phosphorylated residues: {total_phosphorylated} ({perc_phosphorylated:.2f}%)")
        print(f"  Total non-phosphorylated residues: {total_non_phosphorylated} ({perc_non_phosphorylated:.2f}%)")

        unique_lengths, counts = np.unique(all_lengths, return_counts=True)
        print(f"  Most common sequence length: {unique_lengths[np.argmax(counts)]} ({counts.max()} sequences)")
        print(f"  Mean sequence length: {np.mean(all_lengths):.2f}")
        print(f"  Standard deviation of lengths: {np.std(all_lengths):.2f}")

        # Analyze amino acid distribution
        print("\nAmino Acid Statistics:")
        total_amino_acids = sum(amino_acid_counts.values())
        for aa, count in amino_acid_counts.most_common():
            frequency = count / total_amino_acids
            print(f"  {aa}: {count} occurrences ({frequency:.2%})")

        print("Suggested Class Weights for Loss Function:")
        print(f"  Phosphorylated: {weight_phosphorylated:.2f}")
        print(f"  Non-Phosphorylated: {weight_non_phosphorylated:.2f}")

    else:
        print("No sequences found across datasets.")


def validate_position_column(file_path):
    """
    Reads the CSV at file_path and checks that every row in the 'position' column
    contains a list of integers. Raises ValueError if any row fails the check.
    """
    df = pd.read_csv(file_path)

    if 'position' not in df.columns:
        raise ValueError(f"CSV file {file_path} does not have a 'position' column.")

    for idx, row in df.iterrows():
        # Safely parse the position column
        try:
            positions = ast.literal_eval(row['position'])
        except (SyntaxError, ValueError) as e:
            raise ValueError(
                f"Row {idx} in file {file_path}: 'position' cannot be parsed.\n"
                f"Actual value: {row['position']}\nError: {e}"
            )

        if not isinstance(positions, list):
            raise ValueError(
                f"Row {idx} in file {file_path}: 'position' is not a list.\n"
                f"Actual value: {positions}"
            )

        for p in positions:
            if not isinstance(p, int):
                raise ValueError(
                    f"Row {idx} in file {file_path}: 'position' list contains "
                    f"non-integer value: {p} (type {type(p)})"
                )
    print(f"Validation passed: All 'position' entries in {file_path} are lists of integers.")


if __name__ == '__main__':
    # This is the main function to test the dataloader
    print("Testing dataloader")
    import yaml
    from box import Box

    # Load configurations from YAML
    config_file_path = "configs/config.yaml"
    with open(config_file_path, "r") as file:
        config_data = yaml.safe_load(file)

    configs = Box(config_data)

    if not configs.model.use_decoder_block:
        analyze_protein_data(configs)

    # Prepare DataLoaders
    dataloaders = prepare_dataloaders(configs)

    # Access DataLoaders
    train_loader = dataloaders.get("train", None)
    valid_loader = dataloaders.get("valid", None)
    test_loader = dataloaders.get("test", None)

    print("Finished preparing DataLoaders")

    if train_loader:
        print(f"Number of samples in train_loader: {len(train_loader.dataset)}")
        print(f"Number of batches in train_loader: {len(train_loader)}")

    if valid_loader:
        print(f"Number of samples in valid_loader: {len(valid_loader.dataset)}")
        print(f"Number of batches in valid_loader: {len(valid_loader)}")

        # # Test __getitem__ for the 7th item in valid_loader
        # print("\nTesting __getitem__ for the 7th item in valid_loader...")
        # valid_dataset = valid_loader.dataset
        # item = valid_dataset[6]
        #
        # # Print details about the item
        # print("Item Details:")
        # print(f"Input IDs: {item['input_ids']}")
        # print(f"Attention Mask: {item['attention_mask']}")
        # print(f"Labels: {item['labels']}")
        # print(f"Mask: {item['mask']}")

    if test_loader:
        print(f"Number of samples in test_loader: {len(test_loader.dataset)}")
        print(f"Number of batches in test_loader: {len(test_loader)}")