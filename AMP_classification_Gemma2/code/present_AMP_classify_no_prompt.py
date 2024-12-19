import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import PeftModel
import json
import pandas as pd
import re

def load_json(file_path_json):

    with open(file_path_json, 'r') as file:
        json_data = json.load(file)

    # Extract 'instruction', 'input', and 'output' into separate lists
    instructions = [entry.get("instruction", "") for entry in json_data]
    inputs = [entry.get("input", "") for entry in json_data]
    outputs = [entry.get("output", "") for entry in json_data]

    # Display the extracted lists
    instructions[:5], inputs[:5], outputs[:5]  # Displaying only the first 5 for brevity

    return instructions, inputs, outputs


def generate_promoter(promoter):
    chat = [
        { "role": "user", "content": promoter},
    ]
    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=384)
    outputs = tokenizer.decode(outputs[0])
    response = outputs.split('model')[-1].replace('<end_of_turn>\n<eos>', '')
    return response

def make_fasta_file(sequences,path): # 将序列写入fasta文件。

    # 检查文件是否存在
    if not os.path.exists(path):
        # 文件不存在，创建一个新文件
        with open(path, 'w') as f:
            f.write("This is a new file.")

        print(f"文件 {path} 不存在，已创建新文件。")
    else:
        print(f"文件 {path} 已存在。")

    with open(path, 'w') as file:
            for i, seq in enumerate(sequences, start=1):
                seq = seq.upper()
                file.write(f'>Sequence_{i}\n')  # 写入序列标识符
                file.write(f'{seq}\n')  # 写入序列

    print(f"File {path} created and sequences written successfully.")

if __name__ == '__main__':


    mode_path = '../Gemma2/LLM-Research/gemma-2-9b-it'

    for epoch in range(500,9440,10):
        lora_path = f'../output_AMP_classify_model/checkpoint-{epoch}' # 这里改称你的 lora 输出对应 checkpoint 地址

        # 加载tokenizer
        tokenizer = AutoTokenizer.from_pretrained(mode_path)

        # 加载模型
        model = AutoModelForCausalLM.from_pretrained(mode_path, device_map="cuda:0",torch_dtype=torch.bfloat16, trust_remote_code=True).eval()
        model = PeftModel.from_pretrained(model, model_id=lora_path)

        # 加载数据
        instructions, sequences, labels = load_json(file_path_json = '../dataset/AMP_classify_peptide_test_no_prompt.json')
        predictions = []
        patterns = r"\b(?:weak|strong|Weak|Strong)\b"

        labels_save = []

        for promoter,label in zip(sequences, labels):

            promoter = generate_promoter(promoter=promoter)
            matched_patterns = re.search(patterns, promoter)

            if matched_patterns:  # 检查是否有匹配项

                flag = matched_patterns.group()
                predictions.append(flag)
                labels_save.append(label)

                print('prediction flag = ', flag)
                print('actual flag = ', label,'\n')
            
            else:
                print("没有找到匹配项")

            
        df = pd.DataFrame({
        'Prediction': predictions,
        'Actual': labels_save
        })

        # 保存为CSV文件
        df.to_csv(f'../result_AMP_classify/classification_result_prompt_chech_point={epoch}.csv', index=False)

        torch.cuda.empty_cache() 
