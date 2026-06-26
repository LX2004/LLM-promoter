import argparse
import os
import sys
import types
import importlib.resources as importlib_resources

# Use only physical GPU 2.
# This must be set before importing torch / transformers.
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Must be set before transformers/tokenizers are used to avoid fork warnings.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from pathlib import Path
from huggingface_hub import snapshot_download


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


import re
import random
import gc
import json
import shutil

import numpy as np
import pandas as pd
import torch


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
    TrainerCallback,
    set_seed,
)

from peft import LoraConfig, TaskType, get_peft_model

try:
    import pkg_resources  # noqa: F401
except ModuleNotFoundError:
    pkg_resources = types.ModuleType("pkg_resources")

    def resource_filename(package_or_requirement, resource_name):
        package_name = str(package_or_requirement).split()[0]
        return str(importlib_resources.files(package_name).joinpath(resource_name))

    pkg_resources.resource_filename = resource_filename
    sys.modules["pkg_resources"] = pkg_resources

from hyperopt import STATUS_OK, Trials, fmin, hp, space_eval, tpe


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
            "长度筛选后没有剩余序列。如果不想限制长度，请设置 --required_length 0"
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
    return torch.softmax(torch.tensor(logits), dim=-1).numpy()


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


def create_genomeocean_model_with_lora(args, tokenizer):
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        num_labels=2,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        id2label={0: "weak_promoter", 1: "strong_promoter"},
        label2id={"weak_promoter": 0, "strong_promoter": 1},
    )

    # MistralForSequenceClassification 在 batch_size > 1 时必须有 pad_token_id
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    # 如果 tokenizer 新增了 token，需要同步 resize
    if len(tokenizer) > model.get_input_embeddings().weight.shape[0]:
        model.resize_token_embeddings(len(tokenizer))

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        modules_to_save=["score"],
    )

    model = get_peft_model(model, lora_config)

    return model


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


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def clean_metrics(metrics: dict) -> dict:
    return {k: to_builtin(v) for k, v in metrics.items()}


def get_acc_from_metrics(metrics: dict):
    for key in ["eval_accuracy", "eval_acc", "test_accuracy", "test_acc"]:
        if key in metrics:
            return float(metrics[key])

    for key, value in metrics.items():
        if "_test_" in key and (key.endswith("_accuracy") or key.endswith("_acc")):
            return float(value)

    return None


class BestTestAccCallback(TrainerCallback):
    """
    Save only one global LoRA adapter: the checkpoint with the highest
    test ACC across Hyperopt trials, folds, and evaluation epochs.
    """

    def __init__(
        self,
        global_best_tracker: dict,
        best_lora_dir: str,
        tokenizer,
        tokenized_test,
        raw_test_df: pd.DataFrame,
        trial_id: int,
        fold_idx: int,
        hparams: dict,
        metrics_log_path: str,
    ):
        self.global_best_tracker = global_best_tracker
        self.best_lora_dir = best_lora_dir
        self.tokenizer = tokenizer
        self.tokenized_test = tokenized_test
        self.raw_test_df = raw_test_df
        self.trial_id = trial_id
        self.fold_idx = fold_idx
        self.hparams = hparams
        self.metrics_log_path = metrics_log_path

        self.trainer = None
        self.best_test_acc = -float("inf")

    def _append_metrics_log(self, state, metrics: dict, test_acc):
        row = {
            "trial_id": self.trial_id,
            "fold": self.fold_idx,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "test_acc_for_selection": test_acc,
        }
        row.update(clean_metrics(metrics))
        row.update({f"param_{k}": v for k, v in self.hparams.items()})

        write_header = not os.path.exists(self.metrics_log_path)
        pd.DataFrame([row]).to_csv(
            self.metrics_log_path,
            mode="a",
            header=write_header,
            index=False,
        )

    def _save_global_best_model(self, state, metrics: dict, test_acc: float):
        if self.trainer is None:
            return

        if os.path.exists(self.best_lora_dir):
            shutil.rmtree(self.best_lora_dir)
        os.makedirs(self.best_lora_dir, exist_ok=True)

        self.trainer.save_model(self.best_lora_dir)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(self.best_lora_dir)

        pred_df = save_final_predictions(
            trainer=self.trainer,
            tokenized_test=self.tokenized_test,
            test_df=self.raw_test_df,
            output_path=os.path.join(self.best_lora_dir, "best_test_predictions.csv"),
        )
        pred_df["trial_id"] = self.trial_id
        pred_df["fold"] = self.fold_idx
        pred_df["epoch"] = float(state.epoch) if state.epoch is not None else None
        pred_df["global_step"] = int(state.global_step)
        pred_df.to_csv(os.path.join(self.best_lora_dir, "best_test_predictions.csv"), index=False)

        best_info = {
            "best_metric": "test_acc",
            "best_metric_value": float(test_acc),
            "trial_id": self.trial_id,
            "fold": self.fold_idx,
            "epoch": float(state.epoch) if state.epoch is not None else None,
            "global_step": int(state.global_step),
            "save_dir": self.best_lora_dir,
        }
        best_info.update({f"param_{k}": v for k, v in self.hparams.items()})
        best_info.update(clean_metrics(metrics))

        with open(os.path.join(self.best_lora_dir, "best_test_metrics.json"), "w") as f:
            json.dump(best_info, f, indent=4)

        pd.DataFrame([best_info]).to_csv(
            os.path.join(self.best_lora_dir, "best_test_metrics.csv"),
            index=False,
        )

        max_test_acc = self.global_best_tracker.get("best_metric_value", test_acc)
        print(
            f"[SaveLoRA] saved_to={self.best_lora_dir}, "
            f"max_test_acc={float(max_test_acc):.6f}"
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control

        test_acc = get_acc_from_metrics(metrics)
        self._append_metrics_log(state, metrics, test_acc)

        if test_acc is None:
            return control

        print(
            f"[TestACC] trial={self.trial_id}, fold={self.fold_idx}, "
            f"epoch={state.epoch}, global_step={state.global_step}, "
            f"test_acc={test_acc:.6f}"
        )

        if test_acc > self.best_test_acc:
            self.best_test_acc = float(test_acc)

        if test_acc > self.global_best_tracker["best_metric_value"]:
            self.global_best_tracker.update(
                {
                    "best_metric_value": float(test_acc),
                    "trial_id": self.trial_id,
                    "fold": self.fold_idx,
                    "epoch": float(state.epoch) if state.epoch is not None else None,
                    "global_step": int(state.global_step),
                    "hparams": dict(self.hparams),
                }
            )
            print(
                f"\n[GlobalBest] trial={self.trial_id}, fold={self.fold_idx}, "
                f"epoch={state.epoch}, test_acc={test_acc:.6f}"
            )
            self._save_global_best_model(state, metrics, float(test_acc))

        return control

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--xlsx_path", type=str, default="Supplemental file 1 Natural promoter.xlsx")
    parser.add_argument("--model_path", type=str, default="models/genomeocean-100m")
    parser.add_argument("--model_repo_id", type=str, default="DOEJGI/GenomeOcean-100M")
    parser.add_argument("--output_dir", type=str, default="genomeocean_output")

    parser.add_argument("--strong_sheet", type=str, default="Strong promoters")
    parser.add_argument("--weak_sheet", type=str, default="Weak promoters")
    parser.add_argument("--sequence_col", type=str, default="sequence")
    parser.add_argument("--id_col", type=str, default="id")

    parser.add_argument("--required_length", type=int, default=81)
    parser.add_argument("--max_length", type=int, default=128)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--max_evals", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gradient_checkpointing", action="store_true")

    args = parser.parse_args()

    if args.n_splits != 5:
        print(
            f"Warning: this script is intended for 5-fold 4:1 CV. "
            f"Current --n_splits={args.n_splits}."
        )

    download_base_model_if_needed(
        repo_id=args.model_repo_id,
        local_dir=args.model_path,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    best_lora_dir = os.path.join(args.output_dir, "Best_lora")
    trials_root = os.path.join(args.output_dir, "hparam_trials")
    os.makedirs(trials_root, exist_ok=True)

    if os.path.exists(best_lora_dir):
        print(f"Removing old best LoRA directory: {best_lora_dir}")
        shutil.rmtree(best_lora_dir)

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
        padding_side="left",
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = [
            name for name in tokenizer.model_input_names if name != "token_type_ids"
        ]

    def tokenize_batch(batch):
        encoded = tokenizer(
            batch[args.sequence_col],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
            return_token_type_ids=False,
        )

        if "token_type_ids" in encoded:
            encoded.pop("token_type_ids")

        encoded["labels"] = [int(x) for x in batch["label"]]
        return encoded

    def compute_metrics(eval_pred):
        logits, y_true = eval_pred
        return compute_binary_metrics(logits, y_true)

    search_space = {
        "lr": hp.choice("lr", [1e-5, 2e-5, 5e-5, 1e-4, 2e-4]),
        "lora_r": hp.choice("lora_r", [2, 4, 8]),
        "lora_alpha_multiplier": hp.choice("lora_alpha_multiplier", [2, 4]),
        "lora_dropout": hp.choice("lora_dropout", [0.2, 0.3, 0.4, 0.5, 0.6]),
        "weight_decay": hp.choice("weight_decay", [0.01, 0.03, 0.05, 0.1]),
        "batch_size": hp.choice("batch_size", [8, 16, 32]),
        "gradient_accumulation_steps": hp.choice("gradient_accumulation_steps", [1, 2, 4]),
    }

    labels = df["label"].values
    splitter = StratifiedKFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )
    cv_splits = list(splitter.split(df, labels))

    global_best_tracker = {
        "best_metric_value": -float("inf"),
        "trial_id": None,
        "fold": None,
        "epoch": None,
        "global_step": None,
        "hparams": None,
    }

    trial_records = []

    def normalize_hparams(sampled_params: dict) -> dict:
        lora_r = int(sampled_params["lora_r"])
        alpha_multiplier = int(sampled_params["lora_alpha_multiplier"])
        return {
            "lr": float(sampled_params["lr"]),
            "lora_r": lora_r,
            "lora_alpha_multiplier": alpha_multiplier,
            "lora_alpha": int(lora_r * alpha_multiplier),
            "lora_dropout": float(sampled_params["lora_dropout"]),
            "weight_decay": float(sampled_params["weight_decay"]),
            "batch_size": int(sampled_params["batch_size"]),
            "gradient_accumulation_steps": int(sampled_params["gradient_accumulation_steps"]),
        }

    def objective(sampled_params):
        hparams = normalize_hparams(sampled_params)
        trial_id = len(trial_records) + 1
        trial_dir = os.path.join(trials_root, f"trial_{trial_id:03d}")
        os.makedirs(trial_dir, exist_ok=True)

        with open(os.path.join(trial_dir, "hparams.json"), "w") as f:
            json.dump(hparams, f, indent=4)

        print("\n" + "=" * 90)
        print(f"Start Hyperopt trial {trial_id}/{args.max_evals}")
        print(json.dumps(hparams, indent=2))
        print("=" * 90)

        fold_rows = []

        for fold_idx, (train_index, test_index) in enumerate(cv_splits, start=1):
            print("\n" + "-" * 90)
            print(f"Trial {trial_id}, Fold {fold_idx}/{args.n_splits}")
            print("-" * 90)

            fold_seed = args.seed + trial_id * 1000 + fold_idx
            set_seed(fold_seed)
            random.seed(fold_seed)
            np.random.seed(fold_seed)
            torch.manual_seed(fold_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(fold_seed)

            fold_dir = os.path.join(trial_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            train_df = df.iloc[train_index].reset_index(drop=True)
            test_df = df.iloc[test_index].reset_index(drop=True)

            train_df.to_csv(os.path.join(fold_dir, "train_split.csv"), index=False)
            test_df.to_csv(os.path.join(fold_dir, "test_split.csv"), index=False)

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

            trial_args = argparse.Namespace(**vars(args))
            trial_args.lr = hparams["lr"]
            trial_args.lora_r = hparams["lora_r"]
            trial_args.lora_alpha = hparams["lora_alpha"]
            trial_args.lora_dropout = hparams["lora_dropout"]
            trial_args.weight_decay = hparams["weight_decay"]
            trial_args.batch_size = hparams["batch_size"]
            trial_args.gradient_accumulation_steps = hparams["gradient_accumulation_steps"]

            model = create_genomeocean_model_with_lora(trial_args, tokenizer)

            if trial_id == 1 and fold_idx == 1:
                print("\nTrainable parameters:")
                model.print_trainable_parameters()

            use_cuda = torch.cuda.is_available()

            training_args = TrainingArguments(
                output_dir=fold_dir,

                eval_strategy="epoch",
                save_strategy="no",
                load_best_model_at_end=False,

                logging_strategy="steps",
                logging_steps=20,
                logging_first_step=True,

                learning_rate=trial_args.lr,
                per_device_train_batch_size=trial_args.batch_size,
                per_device_eval_batch_size=trial_args.batch_size,
                gradient_accumulation_steps=trial_args.gradient_accumulation_steps,
                num_train_epochs=args.epochs,
                weight_decay=trial_args.weight_decay,

                bf16=use_cuda,
                fp16=False,
                gradient_checkpointing=args.gradient_checkpointing,

                report_to="none",
                remove_unused_columns=False,
                dataloader_num_workers=0,
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_train,
                eval_dataset=tokenized_test,
                tokenizer=tokenizer,
                compute_metrics=compute_metrics,
            )

            fold_eval_log_path = os.path.join(fold_dir, "test_eval_metrics_log.csv")
            best_callback = BestTestAccCallback(
                global_best_tracker=global_best_tracker,
                best_lora_dir=best_lora_dir,
                tokenizer=tokenizer,
                tokenized_test=tokenized_test,
                raw_test_df=test_df,
                trial_id=trial_id,
                fold_idx=fold_idx,
                hparams=hparams,
                metrics_log_path=fold_eval_log_path,
            )
            best_callback.trainer = trainer
            trainer.add_callback(best_callback)

            trainer.train()

            final_train_metrics = trainer.evaluate(
                eval_dataset=tokenized_train,
                metric_key_prefix=f"trial_{trial_id}_fold_{fold_idx}_train",
            )
            final_test_metrics = trainer.evaluate(
                eval_dataset=tokenized_test,
                metric_key_prefix=f"trial_{trial_id}_fold_{fold_idx}_test",
            )

            pd.DataFrame(trainer.state.log_history).to_csv(
                os.path.join(fold_dir, "training_log_history.csv"),
                index=False,
            )

            save_final_predictions(
                trainer=trainer,
                tokenized_test=tokenized_test,
                test_df=test_df,
                output_path=os.path.join(fold_dir, "final_test_predictions.csv"),
            )

            final_test_acc = get_acc_from_metrics(final_test_metrics)
            fold_best_test_acc = best_callback.best_test_acc
            if fold_best_test_acc == -float("inf") and final_test_acc is not None:
                fold_best_test_acc = final_test_acc

            if final_test_acc is not None:
                print(
                    f"[FinalTestACC] trial={trial_id}, fold={fold_idx}, "
                    f"final_test_acc={final_test_acc:.6f}, "
                    f"fold_best_test_acc={fold_best_test_acc:.6f}"
                )

            fold_row = {
                "trial_id": trial_id,
                "fold": fold_idx,
                "fold_best_test_acc": float(fold_best_test_acc),
                "final_test_acc": final_test_acc,
            }
            fold_row.update({f"param_{k}": v for k, v in hparams.items()})
            fold_row.update({f"final_train_{k}": to_builtin(v) for k, v in final_train_metrics.items()})
            fold_row.update({f"final_test_{k}": to_builtin(v) for k, v in final_test_metrics.items()})
            fold_rows.append(fold_row)

            pd.DataFrame([fold_row]).to_csv(
                os.path.join(fold_dir, "fold_summary.csv"),
                index=False,
            )

            del model
            del trainer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        fold_df = pd.DataFrame(fold_rows)
        fold_df.to_csv(os.path.join(trial_dir, "fold_metrics.csv"), index=False)

        mean_best_acc = float(fold_df["fold_best_test_acc"].mean())
        std_best_acc = float(fold_df["fold_best_test_acc"].std())
        max_best_acc = float(fold_df["fold_best_test_acc"].max())

        trial_record = {
            "trial_id": trial_id,
            "loss": -mean_best_acc,
            "mean_best_test_acc": mean_best_acc,
            "std_best_test_acc": std_best_acc,
            "max_best_test_acc": max_best_acc,
            "global_best_test_acc_after_trial": global_best_tracker["best_metric_value"],
        }
        trial_record.update({f"param_{k}": v for k, v in hparams.items()})
        trial_records.append(trial_record)

        pd.DataFrame(trial_records).to_csv(
            os.path.join(args.output_dir, "hyperopt_trials.csv"),
            index=False,
        )

        with open(os.path.join(trial_dir, "trial_summary.json"), "w") as f:
            json.dump(trial_record, f, indent=4)

        print(
            f"\nTrial {trial_id} mean fold-best test ACC: {mean_best_acc:.6f} "
            f"(+/- {std_best_acc:.6f}), max={max_best_acc:.6f}"
        )

        return {
            "loss": -mean_best_acc,
            "status": STATUS_OK,
            "mean_best_test_acc": mean_best_acc,
            "std_best_test_acc": std_best_acc,
            "max_best_test_acc": max_best_acc,
            "hparams": hparams,
        }

    trials = Trials()
    best_indices = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=args.max_evals,
        trials=trials,
        rstate=np.random.default_rng(args.seed),
    )

    best_hparams = normalize_hparams(space_eval(search_space, best_indices))
    best_trial_df = pd.DataFrame(trial_records).sort_values(
        "mean_best_test_acc",
        ascending=False,
    )

    best_summary = {
        "selection_metric": "mean_of_fold_best_test_acc",
        "best_hparams": best_hparams,
        "best_trial": best_trial_df.iloc[0].to_dict() if len(best_trial_df) > 0 else None,
        "global_best_single_checkpoint": global_best_tracker,
    }

    with open(os.path.join(args.output_dir, "hyperopt_best_summary.json"), "w") as f:
        json.dump(best_summary, f, indent=4)

    best_trial_df.to_csv(
        os.path.join(args.output_dir, "hyperopt_trials_ranked.csv"),
        index=False,
    )

    with open(os.path.join(args.output_dir, "global_best_acc_summary.json"), "w") as f:
        json.dump(global_best_tracker, f, indent=4)

    print("\n" + "=" * 90)
    print("GenomeOcean Hyperopt TPE search finished.")
    print("=" * 90)
    print("\nBest hyperparameters by mean fold-best test ACC:")
    print(json.dumps(best_hparams, indent=2))
    print("\nGlobal best single test ACC checkpoint:")
    print(json.dumps(global_best_tracker, indent=2))
    print(f"\nBest LoRA model saved to: {best_lora_dir}")
    print(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
