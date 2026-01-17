from datasets import Dataset
import pandas as pd
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    DataCollatorWithPadding, TrainingArguments, Trainer, TrainerCallback
)
import torch
from peft import LoraConfig, TaskType, get_peft_model
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ===================== Config section =====================

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_PATH = "../Gemma2_model/LLM-Research/gemma-2-9b-it"
TRAIN_JSON_PATH = "../dataset/classify_promoter_train_original_promoter.json"
TEST_JSON_PATH  = "../dataset/classify_promoter_test_original_promoter.json"

LOSS_DIR   = "../loss"
OUTPUT_DIR = "../output_classify_promoter_original_promoter_cls"

NUM_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 256
PER_DEVICE_EVAL_BATCH_SIZE  = 256
GRADIENT_ACCUM_STEPS = 2
LEARNING_RATE = 5e-4
LOGGING_STEPS = 10
SAVE_STEPS = 10

# ===================== Callbacks =====================

class LossSaverCallback(TrainerCallback):
    def __init__(self, save_dir, train_dataset, eval_dataset):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.file_path = os.path.join(self.save_dir, "checkpoint_loss.csv")
        self.trainer = None

    def on_save(self, args, state, control, **kwargs):
        train_metrics = self.trainer.evaluate(
            eval_dataset=self.train_dataset,
            metric_key_prefix="train"
        )
        test_metrics = self.trainer.evaluate(
            eval_dataset=self.eval_dataset,
            metric_key_prefix="test"
        )

        row = {
            "checkpoint": f"checkpoint-{state.global_step}",
            "global_step": state.global_step,
            "epoch": state.epoch,
            "train_loss": train_metrics.get("train_loss"),
            "train_accuracy": train_metrics.get("train_accuracy"),
            "test_loss": test_metrics.get("test_loss"),
            "test_accuracy": test_metrics.get("test_accuracy"),
        }

        df = pd.DataFrame([row])
        write_header = not os.path.exists(self.file_path)
        df.to_csv(self.file_path, mode="a", header=write_header, index=False)
        print(f"[LossSaverCallback] {row}")

class ReduceLROnPlateauCallback(TrainerCallback):
    def __init__(self, optimizer, factor=0.5, patience=10, threshold=0.05):
        self.scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=factor,
            patience=patience,
            threshold=threshold,
            threshold_mode="rel",
            verbose=True,
        )

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" in metrics:
            self.scheduler.step(metrics["eval_loss"])

class RelativeEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=20, min_delta=0.05):
        self.patience = patience
        self.min_delta = min_delta
        self.loss_history = []

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        if "eval_loss" not in metrics:
            return

        self.loss_history.append(metrics["eval_loss"])
        if len(self.loss_history) < self.patience:
            return

        recent = self.loss_history[-self.patience:]
        rel_change = (max(recent) - min(recent)) / max(recent)

        if rel_change < self.min_delta:
            print(
                f"[EarlyStopping] Relative loss change < "
                f"{self.min_delta*100:.1f}% over {self.patience} checkpoints. Stop training."
            )
            control.should_training_stop = True

# ===================== Preprocessing =====================

def process_func(example):
    MAX_LENGTH = 384
    text = (example["instruction"] or "") + (example["input"] or "")
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,
        add_special_tokens=True,
    )
    encoding["labels"] = int(example["label"])
    return encoding

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": (preds == labels).mean()}

# ===================== Main =====================

if __name__ == '__main__':

    os.makedirs(LOSS_DIR, exist_ok=True)

    df_train = pd.read_json(TRAIN_JSON_PATH)
    df_test  = pd.read_json(TEST_JSON_PATH)

    def to_binary_label(x):
        x = str(x).strip().lower()
        if x in ["weak", "0", "weak promoter"]:
            return 0
        elif x in ["strong", "1", "strong promoter"]:
            return 1
        else:
            raise ValueError(f"Unknown label: {x}")

    df_train["label"] = df_train["output"].apply(to_binary_label)
    df_test["label"]  = df_test["output"].apply(to_binary_label)

    ds_train = Dataset.from_pandas(df_train[["instruction", "input", "label"]])
    ds_test  = Dataset.from_pandas(df_test[["instruction", "input", "label"]])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    tokenized_train = ds_train.map(process_func, remove_columns=ds_train.column_names)
    tokenized_test  = ds_test.map(process_func, remove_columns=ds_test.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=[
                "q_proj","k_proj","v_proj","o_proj",
                "gate_proj","up_proj","down_proj"
            ],
        )
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        evaluation_strategy="steps",
        eval_steps=SAVE_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    loss_cb = LossSaverCallback(LOSS_DIR, tokenized_train, tokenized_test)
    loss_cb.trainer = trainer

    trainer.add_callback(loss_cb)
    trainer.add_callback(
        ReduceLROnPlateauCallback(trainer.optimizer, factor=0.5, patience=10, threshold=0.05)
    )
    trainer.add_callback(
        RelativeEarlyStoppingCallback(patience=20, min_delta=0.05)
    )

    trainer.train()
