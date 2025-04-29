import os
# Specify the GPU device to use
os.environ['CUDA_VISIBLE_DEVICES'] = "2"

# Import required libraries
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Data preprocessing function: tokenize each sample and retain the label
def process_func(example):
    tokenized = tokenizer(example['text'], truncation=True, padding='max_length', max_length=200)
    tokenized['label'] = example['label']
    return tokenized

# Evaluation metrics: accuracy, F1 score, precision, recall
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

if __name__ == '__main__':

    # Load the merged JSON dataset
    dataset = load_dataset("json", data_files="../dataset/AMP_classify_peptide_merged.json", split="train")

    # Split the dataset into training and validation sets
    dataset = dataset.train_test_split(test_size=0.2)

    # Load the tokenizer and set pad_token
    tokenizer = AutoTokenizer.from_pretrained('../Gemma2_model/LLM-Research/gemma-2-9b-it')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    # Tokenize training and validation sets
    tokenized_ds = dataset.map(process_func, remove_columns=dataset['train'].column_names if isinstance(dataset, dict) else dataset.column_names)
    print(tokenizer.decode(tokenized_ds['train'][0]['input_ids']))

    # Load Gemma2 model for classification and set number of classes to 2
    model = AutoModelForSequenceClassification.from_pretrained(
        '../Gemma2_model/LLM-Research/gemma-2-9b-it',
        device_map='cuda',
        torch_dtype=torch.bfloat16,
        num_labels=2
    )
    model.enable_input_require_grads()

    # Configure LoRA fine-tuning parameters
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )

    # Set training arguments
    args = TrainingArguments(
        output_dir="../output_AMP_classify_peptide_no_prompt",
        per_device_train_batch_size=32,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=90,
        save_steps=10,
        evaluation_strategy="epoch",
        learning_rate=5e-4,
        save_on_each_node=True,
        gradient_checkpointing=True
    )

    # Combine model with LoRA parameters and prepare for fine-tuning
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # Create the Trainer object
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds['train'],
        eval_dataset=tokenized_ds['test'],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    # Start training
    trainer.train()
