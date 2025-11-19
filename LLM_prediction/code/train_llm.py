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


# ===================== Config section (easy to modify) =====================

# GPU visibility
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Paths to model and dataset
MODEL_PATH = "../Gemma2_model/LLM-Research/gemma-2-9b-it"

TRAIN_JSON_PATH = "../dataset/classify_promoter_train_original_promoter.json"
TEST_JSON_PATH  = "../dataset/classify_promoter_test_original_promoter.json"

# Output directories
LOSS_DIR   = "../loss"
OUTPUT_DIR = "../output_classify_promoter_original_promoter_cls"

# Training hyperparameters
NUM_EPOCHS = 30
PER_DEVICE_TRAIN_BATCH_SIZE = 256
PER_DEVICE_EVAL_BATCH_SIZE  = 256
GRADIENT_ACCUM_STEPS = 2
LEARNING_RATE = 5e-4
LOGGING_STEPS = 10

# How often to save a checkpoint (in steps)
SAVE_STEPS = 10
SAVE_STRATEGY = "steps"   # keep using steps-based checkpointing


# ===================== Callback to save loss/accuracy at each checkpoint =====================

class LossSaverCallback(TrainerCallback):
    """
    Whenever the Trainer saves a checkpoint (on_save):

      1. Run evaluate() on the train_dataset → get train_loss, train_accuracy
      2. Run evaluate() on the eval_dataset  → get test_loss, test_accuracy
      3. Append (checkpoint name, epoch, train_loss, train_accuracy,
         test_loss, test_accuracy) to a CSV file and print it.
    """

    def __init__(self, save_dir, train_dataset, eval_dataset):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.file_path = os.path.join(self.save_dir, "checkpoint_loss.csv")
        self.trainer = None    # Will be injected after the Trainer is created

    def on_save(self, args, state, control, **kwargs):
        assert self.trainer is not None, "LossSaverCallback.trainer has not been set!"

        # 1. Compute train loss / accuracy
        train_metrics = self.trainer.evaluate(
            eval_dataset=self.train_dataset,
            metric_key_prefix="train"
        )
        train_loss = train_metrics.get("train_loss", None)
        train_acc  = train_metrics.get("train_accuracy", None)

        # 2. Compute test loss / accuracy
        test_metrics = self.trainer.evaluate(
            eval_dataset=self.eval_dataset,
            metric_key_prefix="test"
        )
        test_loss = test_metrics.get("test_loss", None)
        test_acc  = test_metrics.get("test_accuracy", None)

        # 3. Record current epoch and checkpoint name
        epoch = state.epoch
        ckpt_name = f"checkpoint-{state.global_step}"

        row = {
            "checkpoint": ckpt_name,
            "global_step": state.global_step,
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc,
        }

        df = pd.DataFrame([row])
        write_header = not os.path.exists(self.file_path)
        df.to_csv(self.file_path, mode="a", header=write_header, index=False)

        print(f"[LossSaverCallback] saved: {row}")


# ===================== Preprocessing function (classification) =====================

def process_func(example):
    """
    Convert one sample into: input_ids, attention_mask, labels (0/1).

    Text = instruction + input from the JSON file.
    Label column is already a binary integer (0/1).
    """
    MAX_LENGTH = 384

    text = (example["instruction"] or "") + (example["input"] or "")

    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False,            # Let DataCollatorWithPadding handle padding
        add_special_tokens=True,
    )

    # label is already an integer 0/1
    encoding["labels"] = int(example["label"])

    return encoding


# ===================== Metrics: compute accuracy =====================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # For binary classification, logits shape is usually (batch_size, 2)
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}


# ===================== Main =====================

if __name__ == '__main__':

    os.makedirs(LOSS_DIR, exist_ok=True)

    # ------------ 1. Load raw train / test JSON ------------
    df_train = pd.read_json(TRAIN_JSON_PATH)
    df_test  = pd.read_json(TEST_JSON_PATH)

    # ------------ 1.1 Map output to 0/1 ------------
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

    df_train_for_hf = df_train[["instruction", "input", "label"]].copy()
    df_test_for_hf  = df_test[["instruction", "input", "label"]].copy()

    ds_train = Dataset.from_pandas(df_train_for_hf)
    ds_test  = Dataset.from_pandas(df_test_for_hf)

    # ------------ 2. Tokenizer & map ------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    tokenized_train = ds_train.map(process_func, remove_columns=ds_train.column_names)
    tokenized_test  = ds_test.map(process_func, remove_columns=ds_test.column_names)

    print(tokenizer.decode(tokenized_train[0]['input_ids']))
    print("Sample label (0=weak,1=strong):", tokenized_train[0]["labels"])

    # ------------ 3. Model & LoRA (binary classification) ------------
    id2label = {0: "weak", 1: "strong"}
    label2id = {"weak": 0, "strong": 1}

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        device_map='cuda',
        torch_dtype=torch.bfloat16,
    )
    model.enable_input_require_grads()

    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # ------------ 4. Training arguments ------------
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUM_STEPS,
        logging_steps=LOGGING_STEPS,
        num_train_epochs=NUM_EPOCHS,
        save_steps=SAVE_STEPS,
        save_strategy=SAVE_STRATEGY,
        learning_rate=LEARNING_RATE,
        save_on_each_node=True,
        gradient_checkpointing=True,
        evaluation_strategy="no",  # All eval is triggered manually in the callback
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ------------ 5. Trainer + callback ------------
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,   # Ensure evaluate() also computes accuracy
    )

    loss_cb = LossSaverCallback(
        save_dir=LOSS_DIR,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
    )
    loss_cb.trainer = trainer
    trainer.add_callback(loss_cb)

    # ------------ 6. Training ------------
    trainer.train()

    # Optional: run a final evaluate on train and test at the end
    train_metrics = trainer.evaluate(
        eval_dataset=tokenized_train,
        metric_key_prefix="train"
    )
    test_metrics = trainer.evaluate(
        eval_dataset=tokenized_test,
        metric_key_prefix="test"
    )

    print("Final Train:", train_metrics)
    print("Final Test :", test_metrics)
