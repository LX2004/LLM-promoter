import argparse
import os

# Use only physical GPU 0.
# This must be set before importing torch / transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Must be set before transformers/tokenizers are used to avoid fork warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from huggingface_hub import snapshot_download

import re
import random
import math
import gc
import json

import numpy as np
import pandas as pd
import torch

from torch.optim.lr_scheduler import LambdaLR

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    matthews_corrcoef,
)

from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model


def has_complete_hf_model(model_dir: str) -> bool:
    """
    Check whether a local Hugging Face model directory already contains
    the key files required by AutoTokenizer.from_pretrained and
    AutoModelForSequenceClassification.from_pretrained.
    """
    model_dir = Path(model_dir)

    if not model_dir.exists() or not model_dir.is_dir():
        return False

    has_config = (model_dir / "config.json").exists()

    has_tokenizer = any(
        (model_dir / name).exists()
        for name in [
            "tokenizer.json",
            "tokenizer.model",
            "vocab.json",
            "merges.txt",
            "special_tokens_map.json",
            "tokenizer_config.json",
        ]
    )

    has_weights = (
        any(model_dir.glob("*.safetensors"))
        or any(model_dir.glob("*.bin"))
        or any(model_dir.glob("pytorch_model*.bin"))
        or any(model_dir.glob("model*.safetensors"))
        or any(model_dir.glob("*.safetensors.index.json"))
        or any(model_dir.glob("pytorch_model*.bin.index.json"))
    )

    return has_config and has_tokenizer and has_weights


def download_base_model_if_needed(repo_id: str, local_dir: str) -> None:
    """
    Download the base model to local_dir only when local_dir does not
    already contain a complete Hugging Face model.
    """
    if has_complete_hf_model(local_dir):
        print(f"Base model already exists. Skipping download: {local_dir}")
        return

    print(f"Base model not found or incomplete: {local_dir}")
    print(f"Downloading base model from Hugging Face: {repo_id}")
    print(f"Target directory: {local_dir}")

    Path(local_dir).mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        resume_download=True,
    )

    if not has_complete_hf_model(local_dir):
        raise RuntimeError(
            "Model download finished, but required model files are still incomplete "
            f"in: {local_dir}"
        )

    print(f"Base model download completed: {local_dir}")


def clean_dna(seq: str) -> str:
    seq = str(seq).upper().strip()
    seq = re.sub(r"\s+", "", seq)
    seq = re.sub(r"[^ACGTN]", "N", seq)
    return seq


def load_promoter_xlsx(
    xlsx_path: str,
    strong_sheet: str = "Strong promoters",
    weak_sheet: str = "Weak promoters",
    sequence_col: str = "sequence",
    id_col: str = "id",
    required_length: int = 81,
):
    strong_df = pd.read_excel(xlsx_path, sheet_name=strong_sheet)
    weak_df = pd.read_excel(xlsx_path, sheet_name=weak_sheet)

    strong_df = strong_df[[id_col, sequence_col]].copy()
    weak_df = weak_df[[id_col, sequence_col]].copy()

    strong_df["label"] = 1
    weak_df["label"] = 0

    strong_df["source_sheet"] = strong_sheet
    weak_df["source_sheet"] = weak_sheet

    df = pd.concat([strong_df, weak_df], axis=0, ignore_index=True)

    df[sequence_col] = df[sequence_col].apply(clean_dna)
    df = df[df[sequence_col].str.len() > 0].copy()

    df["seq_len"] = df[sequence_col].str.len()

    if required_length > 0:
        before_len = len(df)
        df = df[df["seq_len"] == required_length].copy()
        print(f"Keep seq_len == {required_length}, removed: {before_len - len(df)}")

    if len(df) == 0:
        raise ValueError(
            "No sequences remain after length filtering. "
            "If you do not want to restrict sequence length, set --required_length 0."
        )

    conflict_check = df.groupby(sequence_col)["label"].nunique()
    conflict_sequences = conflict_check[conflict_check > 1].index.tolist()

    if len(conflict_sequences) > 0:
        print(f"Warning: found {len(conflict_sequences)} sequences with conflicting labels.")
        print("These conflicting sequences will be removed.")
        df = df[~df[sequence_col].isin(conflict_sequences)].copy()

    before_drop = len(df)
    df = df.drop_duplicates(subset=[sequence_col]).reset_index(drop=True)
    after_drop = len(df)

    print(f"Removed duplicated sequences: {before_drop - after_drop}")

    print("\nLoaded data:")
    print(df["label"].value_counts())

    print("\nSequence length summary:")
    print(df["seq_len"].describe())

    return df


def logits_to_numpy(logits):
    if isinstance(logits, tuple):
        logits = logits[0]
    return np.asarray(logits)


def logits_to_probs(logits):
    logits = logits_to_numpy(logits)
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    return probs


def compute_binary_metrics(logits, labels):
    logits = logits_to_numpy(logits)
    probs = logits_to_probs(logits)
    preds = np.argmax(probs, axis=-1)

    labels = np.asarray(labels)

    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    mcc = matthews_corrcoef(labels, preds)

    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()

    sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc = roc_auc_score(labels, probs[:, 1])
    except ValueError:
        auc = float("nan")

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "sn": sn,
        "sp": sp,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_hf_dataset(df: pd.DataFrame, sequence_col: str = "sequence"):
    keep_cols = [sequence_col, "label"]
    return Dataset.from_pandas(df[keep_cols].reset_index(drop=True))


def create_model_with_lora(args):
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        trust_remote_code=True,
        id2label={0: "weak_promoter", 1: "strong_promoter"},
        label2id={"weak_promoter": 0, "strong_promoter": 1},
    )

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["query", "value"],
        bias="none",
        modules_to_save=["classifier"],
    )

    model = get_peft_model(model, lora_config)
    return model


class EpochStepDecayBestTestTrainer(Trainer):
    """
    Functions:
    1. Every decay_epochs epochs, multiply the learning rate by decay_factor.
    2. After evaluation at each epoch, if the selected test metric reaches a new best,
       save:
       - best_test_model/
       - best_test_predictions.csv
       - best_test_metrics.csv
       - best_test_metrics.json
    """

    def __init__(
        self,
        *args,
        decay_epochs=100,
        decay_factor=0.5,
        best_metric="test_mcc",
        best_greater_is_better=True,
        raw_test_df=None,
        tokenizer_for_saving=None,
        fold_output_dir=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor

        self.best_metric = best_metric
        self.best_greater_is_better = best_greater_is_better
        self.best_metric_value = -float("inf") if best_greater_is_better else float("inf")

        self.raw_test_df = raw_test_df
        self.tokenizer_for_saving = tokenizer_for_saving
        self.fold_output_dir = fold_output_dir

    def create_scheduler(self, num_training_steps: int, optimizer=None):
        if self.lr_scheduler is None:
            if optimizer is None:
                optimizer = self.optimizer

            steps_per_epoch = math.ceil(
                len(self.get_train_dataloader()) / self.args.gradient_accumulation_steps
            )

            def lr_lambda(current_step):
                current_epoch = current_step // max(1, steps_per_epoch)
                decay_times = current_epoch // self.decay_epochs
                return self.decay_factor ** decay_times

            self.lr_scheduler = LambdaLR(optimizer, lr_lambda)

        return self.lr_scheduler

    def _is_better(self, current_value):
        if self.best_greater_is_better:
            return current_value > self.best_metric_value
        return current_value < self.best_metric_value

    def _save_best_test_outputs(self, metrics):
        if self.fold_output_dir is None:
            return

        test_dataset = None

        if isinstance(self.eval_dataset, dict) and "test" in self.eval_dataset:
            test_dataset = self.eval_dataset["test"]

        if test_dataset is None:
            print("Warning: no test dataset found, skip saving best test predictions.")
            return

        best_model_dir = os.path.join(self.fold_output_dir, "best_test_model")
        os.makedirs(best_model_dir, exist_ok=True)

        self.save_model(best_model_dir)

        if self.tokenizer_for_saving is not None:
            self.tokenizer_for_saving.save_pretrained(best_model_dir)

        pred_output = self.predict(test_dataset, metric_key_prefix="best_test")
        logits = logits_to_numpy(pred_output.predictions)
        probs = logits_to_probs(logits)
        preds = np.argmax(probs, axis=-1)

        if self.raw_test_df is not None:
            pred_df = self.raw_test_df.copy()
        else:
            pred_df = pd.DataFrame({"label": pred_output.label_ids})

        pred_df["epoch"] = float(self.state.epoch) if self.state.epoch is not None else None
        pred_df["label"] = pred_output.label_ids

        pred_df["logit_weak_promoter"] = logits[:, 0]
        pred_df["logit_strong_promoter"] = logits[:, 1]

        pred_df["prob_weak_promoter"] = probs[:, 0]
        pred_df["prob_strong_promoter"] = probs[:, 1]

        pred_df["predicted_label"] = preds
        pred_df["predicted_name"] = pred_df["predicted_label"].map(
            {
                0: "weak_promoter",
                1: "strong_promoter",
            }
        )

        pred_df.to_csv(
            os.path.join(self.fold_output_dir, "best_test_predictions.csv"),
            index=False,
        )

        best_metrics = {
            "best_epoch": float(self.state.epoch) if self.state.epoch is not None else None,
            "best_metric": self.best_metric,
            "best_metric_value": self.best_metric_value,
        }

        for k, v in metrics.items():
            if isinstance(v, np.generic):
                v = v.item()
            best_metrics[k] = v

        with open(os.path.join(self.fold_output_dir, "best_test_metrics.json"), "w") as f:
            json.dump(best_metrics, f, indent=4)

        pd.DataFrame([best_metrics]).to_csv(
            os.path.join(self.fold_output_dir, "best_test_metrics.csv"),
            index=False,
        )

        print(
            f"\nSaved new best test model at epoch {best_metrics['best_epoch']}, "
            f"{self.best_metric} = {self.best_metric_value:.6f}"
        )

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )

        if self.best_metric.startswith("eval_"):
            metric_key = self.best_metric
        else:
            metric_key = f"eval_{self.best_metric}"

        if metric_key in metrics:
            current_value = metrics[metric_key]

            if current_value is not None and not np.isnan(current_value):
                if self._is_better(current_value):
                    self.best_metric_value = current_value
                    self._save_best_test_outputs(metrics)

        return metrics


def save_final_predictions(trainer, tokenized_test, test_df, output_path):
    pred_output = trainer.predict(tokenized_test)
    logits = logits_to_numpy(pred_output.predictions)
    probs = logits_to_probs(logits)
    preds = np.argmax(probs, axis=-1)

    pred_df = test_df.copy()
    pred_df["label"] = pred_output.label_ids

    pred_df["logit_weak_promoter"] = logits[:, 0]
    pred_df["logit_strong_promoter"] = logits[:, 1]

    pred_df["prob_weak_promoter"] = probs[:, 0]
    pred_df["prob_strong_promoter"] = probs[:, 1]

    pred_df["predicted_label"] = preds
    pred_df["predicted_name"] = pred_df["predicted_label"].map(
        {
            0: "weak_promoter",
            1: "strong_promoter",
        }
    )

    pred_df.to_csv(output_path, index=False)

    return pred_df


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--xlsx_path", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="models/nt-v2-50m")
    parser.add_argument(
        "--model_repo_id",
        type=str,
        default="InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    )
    parser.add_argument("--output_dir", type=str, default="nt_output")

    parser.add_argument("--strong_sheet", type=str, default="Strong promoters")
    parser.add_argument("--weak_sheet", type=str, default="Weak promoters")
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--id_col", type=str, default="id")

    parser.add_argument("--required_length", type=int, default=81)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--lr_decay_epochs", type=int, default=100)
    parser.add_argument("--lr_decay_factor", type=float, default=0.5)

    parser.add_argument("--best_metric", type=str, default="test_mcc")

    args = parser.parse_args()

    download_base_model_if_needed(
        repo_id=args.model_repo_id,
        local_dir=args.model_path,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True

    df = load_promoter_xlsx(
        xlsx_path=args.xlsx_path,
        strong_sheet=args.strong_sheet,
        weak_sheet=args.weak_sheet,
        sequence_col=args.sequence_col,
        id_col=args.id_col,
        required_length=args.required_length,
    )

    df.to_csv(
        os.path.join(args.output_dir, "all_cleaned_data.csv"),
        index=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch[args.sequence_col],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        encoded["labels"] = [int(x) for x in batch["label"]]
        return encoded

    labels = df["label"].values

    skf = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_final_metrics = []
    fold_best_metrics = []
    all_final_test_predictions = []
    all_best_test_predictions = []

    for fold_idx, (train_index, test_index) in enumerate(skf.split(df, labels), start=1):
        print("\n" + "=" * 90)
        print(f"Start Fold {fold_idx}/{args.n_splits}")
        print("=" * 90)

        fold_output_dir = os.path.join(args.output_dir, f"fold_{fold_idx}")
        os.makedirs(fold_output_dir, exist_ok=True)

        train_df = df.iloc[train_index].reset_index(drop=True)
        test_df = df.iloc[test_index].reset_index(drop=True)

        print(f"\nFold {fold_idx} train size: {len(train_df)}")
        print(train_df["label"].value_counts())

        print(f"\nFold {fold_idx} test size: {len(test_df)}")
        print(test_df["label"].value_counts())

        train_df.to_csv(
            os.path.join(fold_output_dir, "train_split.csv"),
            index=False,
        )

        test_df.to_csv(
            os.path.join(fold_output_dir, "test_split.csv"),
            index=False,
        )

        train_dataset = build_hf_dataset(train_df, sequence_col=args.sequence_col)
        test_dataset = build_hf_dataset(test_df, sequence_col=args.sequence_col)

        tokenized_train = train_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=train_dataset.column_names,
        )

        tokenized_test = test_dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=test_dataset.column_names,
        )

        model = create_model_with_lora(args)

        if fold_idx == 1:
            print("\nTrainable parameters:")
            model.print_trainable_parameters()

        def compute_metrics(eval_pred):
            logits, y_true = eval_pred
            return compute_binary_metrics(logits, y_true)

        use_cuda = torch.cuda.is_available()

        training_args = TrainingArguments(
            output_dir=fold_output_dir,

            eval_strategy="epoch",
            save_strategy="no",
            load_best_model_at_end=False,

            logging_strategy="steps",
            logging_steps=20,
            logging_first_step=True,

            learning_rate=args.lr,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,

            bf16=use_cuda,
            fp16=False,

            report_to="none",
            remove_unused_columns=False,

            # Set this to 0 to avoid potential dataloader deadlocks in notebooks/remote sessions.
            dataloader_num_workers=0,
        )

        trainer = EpochStepDecayBestTestTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset={
                "train": tokenized_train,
                "test": tokenized_test,
            },
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,

            decay_epochs=args.lr_decay_epochs,
            decay_factor=args.lr_decay_factor,

            best_metric=args.best_metric,
            best_greater_is_better=True,
            raw_test_df=test_df,
            tokenizer_for_saving=tokenizer,
            fold_output_dir=fold_output_dir,
        )

        trainer.train()

        print(f"\nFold {fold_idx} final train result:")
        train_result = trainer.evaluate(
            eval_dataset=tokenized_train,
            metric_key_prefix=f"fold_{fold_idx}_train",
        )
        print(train_result)

        print(f"\nFold {fold_idx} final test result:")
        test_result = trainer.evaluate(
            eval_dataset=tokenized_test,
            metric_key_prefix=f"fold_{fold_idx}_test",
        )
        print(test_result)

        final_model_dir = os.path.join(fold_output_dir, "final_model")
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        log_history_df = pd.DataFrame(trainer.state.log_history)
        log_history_df.to_csv(
            os.path.join(fold_output_dir, "training_log_history.csv"),
            index=False,
        )

        final_pred_df = save_final_predictions(
            trainer=trainer,
            tokenized_test=tokenized_test,
            test_df=test_df,
            output_path=os.path.join(fold_output_dir, "final_test_predictions.csv"),
        )
        final_pred_df["fold"] = fold_idx
        all_final_test_predictions.append(final_pred_df)

        best_pred_path = os.path.join(fold_output_dir, "best_test_predictions.csv")
        if os.path.exists(best_pred_path):
            best_pred_df = pd.read_csv(best_pred_path)
            best_pred_df["fold"] = fold_idx
            all_best_test_predictions.append(best_pred_df)

        best_metrics_path = os.path.join(fold_output_dir, "best_test_metrics.json")
        if os.path.exists(best_metrics_path):
            with open(best_metrics_path, "r") as f:
                best_metrics = json.load(f)

            best_metrics["fold"] = fold_idx
            fold_best_metrics.append(best_metrics)

        final_row = {
            "fold": fold_idx,

            "final_train_acc": train_result.get(f"fold_{fold_idx}_train_acc"),
            "final_train_sp": train_result.get(f"fold_{fold_idx}_train_sp"),
            "final_train_sn": train_result.get(f"fold_{fold_idx}_train_sn"),
            "final_train_auc": train_result.get(f"fold_{fold_idx}_train_auc"),
            "final_train_f1": train_result.get(f"fold_{fold_idx}_train_f1"),
            "final_train_mcc": train_result.get(f"fold_{fold_idx}_train_mcc"),

            "final_test_acc": test_result.get(f"fold_{fold_idx}_test_acc"),
            "final_test_sp": test_result.get(f"fold_{fold_idx}_test_sp"),
            "final_test_sn": test_result.get(f"fold_{fold_idx}_test_sn"),
            "final_test_auc": test_result.get(f"fold_{fold_idx}_test_auc"),
            "final_test_f1": test_result.get(f"fold_{fold_idx}_test_f1"),
            "final_test_mcc": test_result.get(f"fold_{fold_idx}_test_mcc"),

            "final_test_precision": test_result.get(f"fold_{fold_idx}_test_precision"),
            "final_test_recall": test_result.get(f"fold_{fold_idx}_test_recall"),
            "final_test_tn": test_result.get(f"fold_{fold_idx}_test_tn"),
            "final_test_fp": test_result.get(f"fold_{fold_idx}_test_fp"),
            "final_test_fn": test_result.get(f"fold_{fold_idx}_test_fn"),
            "final_test_tp": test_result.get(f"fold_{fold_idx}_test_tp"),
        }

        fold_final_metrics.append(final_row)

        pd.DataFrame([final_row]).to_csv(
            os.path.join(fold_output_dir, "final_metrics.csv"),
            index=False,
        )

        del model
        del trainer
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    final_metrics_df = pd.DataFrame(fold_final_metrics)
    final_metrics_df.to_csv(
        os.path.join(args.output_dir, "five_fold_final_metrics.csv"),
        index=False,
    )

    if len(fold_best_metrics) > 0:
        best_metrics_df = pd.DataFrame(fold_best_metrics)
        best_metrics_df.to_csv(
            os.path.join(args.output_dir, "five_fold_best_test_metrics.csv"),
            index=False,
        )
    else:
        best_metrics_df = pd.DataFrame()

    if len(all_final_test_predictions) > 0:
        all_final_pred_df = pd.concat(all_final_test_predictions, axis=0, ignore_index=True)
        all_final_pred_df.to_csv(
            os.path.join(args.output_dir, "five_fold_final_test_predictions.csv"),
            index=False,
        )

    if len(all_best_test_predictions) > 0:
        all_best_pred_df = pd.concat(all_best_test_predictions, axis=0, ignore_index=True)
        all_best_pred_df.to_csv(
            os.path.join(args.output_dir, "five_fold_best_test_predictions.csv"),
            index=False,
        )

    summary_rows = []

    if not best_metrics_df.empty:
        best_metric_cols = [
            "eval_test_acc",
            "eval_test_sp",
            "eval_test_sn",
            "eval_test_auc",
            "eval_test_f1",
            "eval_test_mcc",
            "eval_test_precision",
            "eval_test_recall",
        ]

        for col in best_metric_cols:
            if col in best_metrics_df.columns:
                summary_rows.append(
                    {
                        "type": "best_test",
                        "metric": col,
                        "mean": best_metrics_df[col].mean(),
                        "std": best_metrics_df[col].std(),
                    }
                )

    final_metric_cols = [
        "final_test_acc",
        "final_test_sp",
        "final_test_sn",
        "final_test_auc",
        "final_test_f1",
        "final_test_mcc",
        "final_test_precision",
        "final_test_recall",
    ]

    for col in final_metric_cols:
        if col in final_metrics_df.columns:
            summary_rows.append(
                {
                    "type": "final_test",
                    "metric": col,
                    "mean": final_metrics_df[col].mean(),
                    "std": final_metrics_df[col].std(),
                }
            )

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        os.path.join(args.output_dir, "five_fold_summary_mean_std.csv"),
        index=False,
    )

    print("\n" + "=" * 90)
    print("Five-fold cross-validation finished.")
    print("=" * 90)

    print("\nFinal per-fold metrics:")
    print(final_metrics_df)

    if not best_metrics_df.empty:
        print("\nBest-test per-fold metrics:")
        print(best_metrics_df)

    print("\nMean ± std:")
    print(summary_df)

    print(f"\nAll results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
    