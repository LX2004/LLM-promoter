import os
os.environ['CUDA_VISIBLE_DEVICES'] = "2"
from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, TrainingArguments, Trainer, GenerationConfig
import torch
from peft import LoraConfig, TaskType, get_peft_model


def process_func(example):
    MAX_LENGTH = 384    # Llama分词器会将一个中文字切分为多个token，因此需要放开一些最大长度，保证数据的完整性
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"<bos><start_of_turn>user\n{example['instruction'] + example['input']}<end_of_turn>\n<start_of_turn>model\n", add_special_tokens=False)  # add_special_tokens 不在开头加 special_tokens
    response = tokenizer(f"{example['output']}<end_of_turn>\n", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]  # 因为eos token咱们也是要关注的所以 补充为1
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]  
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

if __name__ == '__main__':

    # 将JSON文件转换为CSV文件
    df = pd.read_json('../dataset/AMP_classify_peptide_train_no_prompt.json')
    ds = Dataset.from_pandas(df)

    tokenizer = AutoTokenizer.from_pretrained('../Gemma2/LLM-Research/gemma-2-9b-it')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = 'right'

    tokenized_id = ds.map(process_func, remove_columns=ds.column_names)
    print(tokenizer.decode(tokenized_id[0]['input_ids']))

    model = AutoModelForCausalLM.from_pretrained('../Gemma2_model/LLM-Research/gemma-2-9b-it', device_map='cuda', torch_dtype=torch.bfloat16,)
    # print('model = ',model)
    model.enable_input_require_grads() # 开启梯度检查点时，要执行该方法

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM, 
        # task_type=TaskType.SEQ_CLS, 
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 'gate_proj', 'up_proj', 'down_proj'],
        inference_mode=False, # 训练模式
        r=8, # Lora 秩
        lora_alpha=32, # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.1# Dropout 比例
    )
    args = TrainingArguments(
    output_dir="../output_AMP_classify_model",
    per_device_train_batch_size=32,
    gradient_accumulation_steps=4,
    logging_steps=10,
    num_train_epochs=90,
    save_steps=10, # 为了快速演示，这里设置10，建议你设置成100
    learning_rate=5e-4,
    save_on_each_node=True,
    gradient_checkpointing=True
)
    
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_id,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
    
    trainer.train()