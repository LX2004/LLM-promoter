import os
import json
import random

# ----------------- Config -----------------
# Paths to the original FASTA files
STRONG_FASTA = "../data/strong_promoters.fasta"
WEAK_FASTA   = "../data/weak_promoters.fasta"

# Output directory & filenames (match your training script)
OUTPUT_DIR = "../dataset"
TRAIN_JSON = "train_original_promoter.json"
TEST_JSON  = "test_original_promoter.json"

# Train / test split ratio
TRAIN_RATIO = 0.8

# Random seed for reproducible splits
RANDOM_SEED = 42

# Whether to include a natural-language instruction (prompt)
USE_PROMPT = True  # Set to False for empty instruction

PROMPT_TEXT = "Determine the strength of Escherichia coli promoters."


# ----------------- FASTA parser -----------------
def parse_fasta(filepath):
    """
    Simple FASTA parser.

    Args:
        filepath (str): Path to a FASTA file.

    Returns:
        List[str]: List of sequences (A/T/C/G...), with multi-line
                   sequences concatenated and uppercased.
    """
    sequences = []
    seq_lines = []

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                # New header: flush existing sequence if any
                if seq_lines:
                    sequences.append("".join(seq_lines).upper())
                    seq_lines = []
            else:
                seq_lines.append(line)

        # Flush the last sequence at EOF
        if seq_lines:
            sequences.append("".join(seq_lines).upper())

    return sequences


# ----------------- Main logic -----------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)

    # 1. Load strong / weak promoter sequences from FASTA
    strong_seqs = parse_fasta(STRONG_FASTA)
    weak_seqs   = parse_fasta(WEAK_FASTA)

    print(f"Loaded {len(strong_seqs)} strong promoters.")
    print(f"Loaded {len(weak_seqs)} weak promoters.")

    # Decide the instruction text according to the flag
    instruction_text = PROMPT_TEXT if USE_PROMPT else ""

    # 2. Convert to unified JSON sample format
    strong_samples = [
        {
            "instruction": instruction_text,
            "input": seq,
            "output": "strong",
        }
        for seq in strong_seqs
    ]

    weak_samples = [
        {
            "instruction": instruction_text,
            "input": seq,
            "output": "weak",
        }
        for seq in weak_seqs
    ]

    # 3. Stratified split: split strong and weak separately, then merge
    def split_by_ratio(samples, train_ratio):
        n = len(samples)
        idx = list(range(n))
        random.shuffle(idx)
        split = int(n * train_ratio)
        train_idx = idx[:split]
        test_idx = idx[split:]
        train_part = [samples[i] for i in train_idx]
        test_part  = [samples[i] for i in test_idx]
        return train_part, test_part

    strong_train, strong_test = split_by_ratio(strong_samples, TRAIN_RATIO)
    weak_train,   weak_test   = split_by_ratio(weak_samples, TRAIN_RATIO)

    train_data = strong_train + weak_train
    test_data  = strong_test  + weak_test

    # Shuffle inside train and test for randomness
    random.shuffle(train_data)
    random.shuffle(test_data)

    print(f"Train size: {len(train_data)} (strong: {len(strong_train)}, weak: {len(weak_train)})")
    print(f"Test  size: {len(test_data)}  (strong: {len(strong_test)},  weak: {len(weak_test)})")

    # 4. Save as JSON (list of dicts)
    train_path = os.path.join(OUTPUT_DIR, TRAIN_JSON)
    test_path  = os.path.join(OUTPUT_DIR, TEST_JSON)

    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=4)

    with open(test_path, "w") as f:
        json.dump(test_data, f, indent=4)

    print(f"Saved train json to: {train_path}")
    print(f"Saved test  json to: {test_path}")


if __name__ == "__main__":
    main()
