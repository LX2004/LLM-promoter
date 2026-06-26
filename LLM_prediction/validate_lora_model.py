import argparse
import gc
import os
import warnings
from dataclasses import dataclass
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import logging as transformers_logging

warnings.filterwarnings("ignore")
transformers_logging.set_verbosity_error()

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_VALIDATION_DIR = PROJECT_DIR / "validation_data"
DEFAULT_TRAINED_MODEL_DIR = PROJECT_DIR / "trained_model"


@dataclass
class EvalConfig:
    key: str
    name: str
    base_model: str
    lora_dir: str
    test_csv: str
    max_length: int
    batch_size: int
    padding_side: str
    use_prompt: bool = False
    prompt_text: str = ""
    use_device_map: bool = False
    use_bfloat16: bool = False


def resolve_path(path_value):
    if path_value is None:
        return None
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = PROJECT_DIR / path
    return str(path)


def require_path(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")


def require_lora_dir(lora_dir):
    require_path(lora_dir, "LoRA directory")
    lora_path = Path(lora_dir)
    if not (lora_path / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json not found in: {lora_dir}")
    if not (lora_path / "adapter_model.safetensors").exists() and not (lora_path / "adapter_model.bin").exists():
        raise FileNotFoundError(f"adapter_model.safetensors or adapter_model.bin not found in: {lora_dir}")


def has_tokenizer_files(path):
    names = [
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "vocab.json",
        "vocab.txt",
        "merges.txt",
        "special_tokens_map.json",
    ]
    return any((Path(path) / name).exists() for name in names)


def load_test_csv(config, sequence_col, label_col):
    require_path(config.test_csv, f"{config.name} test CSV")
    df = pd.read_csv(config.test_csv)
    missing = [col for col in [sequence_col, label_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {config.test_csv}")
    return df


def build_texts(df, sequence_col, config):
    seqs = df[sequence_col].astype(str).tolist()
    if not config.use_prompt:
        return seqs
    prompt = config.prompt_text.strip()
    return [f"{prompt} {seq}" for seq in seqs]


def load_tokenizer(config):
    tokenizer_path = config.lora_dir if has_tokenizer_files(config.lora_dir) else config.base_model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = config.padding_side

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if hasattr(tokenizer, "model_input_names"):
        tokenizer.model_input_names = [name for name in tokenizer.model_input_names if name != "token_type_ids"]

    return tokenizer


def load_model(config, tokenizer):
    require_path(config.base_model, f"{config.name} base model")
    require_lora_dir(config.lora_dir)

    dtype = torch.bfloat16 if config.use_bfloat16 and torch.cuda.is_available() else torch.float32
    kwargs = {
        "num_labels": 2,
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "id2label": {0: "weak_promoter", 1: "strong_promoter"},
        "label2id": {"weak_promoter": 0, "strong_promoter": 1},
    }
    if config.key.startswith("gemma"):
        kwargs["problem_type"] = "single_label_classification"
    if config.use_device_map and torch.cuda.is_available():
        kwargs["device_map"] = {"": 0}

    base_model = AutoModelForSequenceClassification.from_pretrained(config.base_model, **kwargs)

    if len(tokenizer) > base_model.get_input_embeddings().weight.shape[0]:
        base_model.resize_token_embeddings(len(tokenizer))

    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    model = PeftModel.from_pretrained(base_model, config.lora_dir, is_trainable=False)
    model.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not (config.use_device_map and torch.cuda.is_available()):
        model.to(device)

    return model, device


@torch.no_grad()
def evaluate_acc(config, df, sequence_col, label_col):
    tokenizer = load_tokenizer(config)
    model, device = load_model(config, tokenizer)

    texts = build_texts(df, sequence_col, config)
    labels = df[label_col].astype(int).tolist()
    preds = []

    for start in range(0, len(texts), config.batch_size):
        batch_texts = texts[start:start + config.batch_size]
        encoded = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=config.max_length,
            return_tensors="pt",
            return_token_type_ids=False,
            add_special_tokens=True,
        )
        if "token_type_ids" in encoded:
            encoded.pop("token_type_ids")
        encoded = {key: value.to(device) for key, value in encoded.items()}
        logits = model(**encoded).logits.detach().float()
        preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

    acc = accuracy_score(labels, preds)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return float(acc), len(labels)


def build_configs(args):
    validation_dir = Path(resolve_path(args.validation_dir))
    trained_model_dir = Path(resolve_path(args.trained_model_dir))

    return {
        "gemma": EvalConfig(
            key="gemma",
            name="Gemma2-9B-NoPrompt",
            base_model=resolve_path(args.gemma_base_model_path),
            lora_dir=resolve_path(args.gemma_lora_dir) if args.gemma_lora_dir else str(trained_model_dir / "gemma_lora"),
            test_csv=resolve_path(args.gemma_test_csv) if args.gemma_test_csv else str(validation_dir / "gemma_test.csv"),
            max_length=args.gemma_max_length,
            batch_size=args.gemma_batch_size,
            padding_side="right",
            use_prompt=False,
            use_device_map=args.gemma_use_device_map,
            use_bfloat16=args.gemma_use_bfloat16,
        ),
        "gemma_prompt": EvalConfig(
            key="gemma_prompt",
            name="Gemma2-9B-Prompt",
            base_model=resolve_path(args.gemma_prompt_base_model_path),
            lora_dir=resolve_path(args.gemma_prompt_lora_dir) if args.gemma_prompt_lora_dir else str(trained_model_dir / "gemma_lora_prompt"),
            test_csv=resolve_path(args.gemma_prompt_test_csv) if args.gemma_prompt_test_csv else str(validation_dir / "gemma_prompt_test.csv"),
            max_length=args.gemma_prompt_max_length,
            batch_size=args.gemma_prompt_batch_size,
            padding_side="right",
            use_prompt=True,
            prompt_text=args.gemma_prompt_text,
            use_device_map=args.gemma_prompt_use_device_map,
            use_bfloat16=args.gemma_prompt_use_bfloat16,
        ),
        "genomeocean": EvalConfig(
            key="genomeocean",
            name="GenomeOcean",
            base_model=resolve_path(args.genomeocean_base_model_path),
            lora_dir=resolve_path(args.genomeocean_lora_dir) if args.genomeocean_lora_dir else str(trained_model_dir / "go_lora"),
            test_csv=resolve_path(args.genomeocean_test_csv) if args.genomeocean_test_csv else str(validation_dir / "go_test.csv"),
            max_length=args.genomeocean_max_length,
            batch_size=args.genomeocean_batch_size,
            padding_side="left",
            use_bfloat16=args.genomeocean_use_bfloat16,
        ),
        "nt": EvalConfig(
            key="nt",
            name="NT-v2-50M",
            base_model=resolve_path(args.nt_base_model_path),
            lora_dir=resolve_path(args.nt_lora_dir) if args.nt_lora_dir else str(trained_model_dir / "nt_lora"),
            test_csv=resolve_path(args.nt_test_csv) if args.nt_test_csv else str(validation_dir / "nt_test.csv"),
            max_length=args.nt_max_length,
            batch_size=args.nt_batch_size,
            padding_side="right",
            use_bfloat16=args.nt_use_bfloat16,
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_model", default="all", choices=["all", "gemma", "gemma_no_prompt", "gemma_prompt", "genomeocean", "nt"])
    parser.add_argument("--validation_dir", default=str(DEFAULT_VALIDATION_DIR))
    parser.add_argument("--trained_model_dir", default=str(DEFAULT_TRAINED_MODEL_DIR))
    parser.add_argument("--sequence_col", default="sequence")
    parser.add_argument("--label_col", default="label")

    parser.add_argument("--gemma_base_model_path", default="models/gemma-2-9b-it")
    parser.add_argument("--gemma_lora_dir", default=None)
    parser.add_argument("--gemma_test_csv", default=None)
    parser.add_argument("--gemma_max_length", type=int, default=384)
    parser.add_argument("--gemma_batch_size", type=int, default=1)
    parser.add_argument("--gemma_use_device_map", action="store_true", default=True)
    parser.add_argument("--no_gemma_device_map", action="store_false", dest="gemma_use_device_map")
    parser.add_argument("--gemma_use_bfloat16", action="store_true", default=True)
    parser.add_argument("--no_gemma_bfloat16", action="store_false", dest="gemma_use_bfloat16")

    parser.add_argument("--gemma_prompt_base_model_path", default="models/gemma-2-9b-it")
    parser.add_argument("--gemma_prompt_lora_dir", default=None)
    parser.add_argument("--gemma_prompt_test_csv", default=None)
    parser.add_argument("--gemma_prompt_max_length", type=int, default=384)
    parser.add_argument("--gemma_prompt_batch_size", type=int, default=8)
    parser.add_argument("--gemma_prompt_text", default="Determine the strength of Escherichia coli promoters.")
    parser.add_argument("--gemma_prompt_use_device_map", action="store_true", default=True)
    parser.add_argument("--no_gemma_prompt_device_map", action="store_false", dest="gemma_prompt_use_device_map")
    parser.add_argument("--gemma_prompt_use_bfloat16", action="store_true", default=True)
    parser.add_argument("--no_gemma_prompt_bfloat16", action="store_false", dest="gemma_prompt_use_bfloat16")

    parser.add_argument("--genomeocean_base_model_path", default="models/genomeocean-100m")
    parser.add_argument("--genomeocean_lora_dir", default=None)
    parser.add_argument("--genomeocean_test_csv", default=None)
    parser.add_argument("--genomeocean_max_length", type=int, default=128)
    parser.add_argument("--genomeocean_batch_size", type=int, default=64)
    parser.add_argument("--genomeocean_use_bfloat16", action="store_true", default=False)

    parser.add_argument("--nt_base_model_path", default="models/nt-v2-50m")
    parser.add_argument("--nt_lora_dir", default=None)
    parser.add_argument("--nt_test_csv", default=None)
    parser.add_argument("--nt_max_length", type=int, default=128)
    parser.add_argument("--nt_batch_size", type=int, default=64)
    parser.add_argument("--nt_use_bfloat16", action="store_true", default=False)
    return parser.parse_args()


def main():
    args = parse_args()
    configs = build_configs(args)

    if args.run_model == "all":
        selected = ["gemma", "gemma_prompt", "genomeocean", "nt"]
    elif args.run_model == "gemma_no_prompt":
        selected = ["gemma"]
    else:
        selected = [args.run_model]

    results = []
    for key in selected:
        config = configs[key]
        try:
            df = load_test_csv(config, args.sequence_col, args.label_col)
            acc, n = evaluate_acc(config, df, args.sequence_col, args.label_col)
            results.append((config.name, n, acc, config.test_csv))
            print(f"{config.name} Test ACC: {acc:.6f}", flush=True)
        except Exception as exc:
            results.append((config.name, None, None, config.test_csv))
            print(f"{config.name} FAILED: {exc}", flush=True)

    print("\nFinal ACC summary", flush=True)
    for name, n, acc, test_csv in results:
        if acc is None:
            print(f"{name}: FAILED | test={test_csv}", flush=True)
        else:
            print(f"{name}: ACC={acc:.6f} | samples={n} | test={test_csv}", flush=True)


if __name__ == "__main__":
    main()
