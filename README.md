# A hybrid large language model and diffusion framework for rational promoter design.

# Framework

<img width="923" height="709" alt="image" src="https://github.com/user-attachments/assets/79fc1283-e80b-4380-80a2-ee6d3e46563b" />

### Strength Prediction  
The LLM (Gemma2) predicts the strength (strong or weak) of an input sequence provided by the user. If the prediction is incorrect, the model is fine-tuned using the LoRA method to improve accuracy.  

### Pseudo-sequence Mutation  
The input sequence undergoes mutations to generate multiple variants, enabling the analysis of different promoter regions' contributions to strength.  

### Core Region Identification  
By evaluating the impact of mutations on predicted strength, the core region of the sequence—where variations significantly affect strength—is identified.  

### Promoter Synthesis  
Based on the identified core region, a diffusion model reconstructs non-core regions, ultimately generating a complete promoter sequence.  

**Tip:** The high-throughput promoter biological experiment data are stored in  
`Supplemental file 3 biological experiment results.xlsx`.

# For classification task
For LLM environment configuration, please refer to:
- [Self-LLM Gemma2 LoRA Fine-tuning Guide](https://github.com/datawhalechina/self-llm/blob/master/models/Gemma2/04-Gemma-2-9b-it%20peft%20lora%E5%BE%AE%E8%B0%83.md)  
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main)
- **Nucleotide Transformer**: [instadeepai/nucleotide-transformer](https://github.com/instadeepai/nucleotide-transformer)
- **GenomeOcean**: [jgi-genomeocean/genomeocean](https://github.com/jgi-genomeocean/genomeocean)

Alternatively, you can run the following code to create a virtual environment.

### Step 1: Ensure Conda is Installed  
Make sure you have Anaconda or Miniconda installed. You can check by running:  

```bash
conda --version
```
### Step 2: Create the Environment
```
git clone https://github.com/LX2004/LLM-promoter.git
cd LLM-promoter
conda env create -f Gemma2_environment.yml
```
### Step 3: Activate the Environment
```
conda activate nt_lora
```

### Step 4: Promoter classification

Before running the code, please make sure your project directory is organized as follows:

```text
LLM_prediction/
├── train_gemma_opt.py
├── train_gemma_prompt_opt.py
├── train_genomeocean_opt.py
├── train_nt_opt.py
├── validate_lora_model.py
├── Supplemental file 1 Natural promoter.xlsx
├── env.yml
├── models/
│   ├── gemma-2-9b-it/
│   ├── genomeocean-100m/
│   └── nt-v2-50m/
├── trained_model/
│   ├── gemma_lora/
│   ├── gemma_lora_prompt/
│   ├── go_lora/
│   └── nt_lora/
└── validation_data/
    ├── gemma_test.csv
    ├── gemma_prompt_test.csv
    ├── go_test.csv
    └── nt_test.csv
```
    
The `models/` directory should contain the base large language models:

```text
models/
├── gemma-2-9b-it/
├── genomeocean-100m/
└── nt-v2-50m/
```

If a model is missing, the script will try to download it automatically. Download errors are usually caused by not logging into Hugging Face or lacking model access. Models can also be downloaded manually and placed in the corresponding directory.

To reproduce the reported results, download the trained model weights from the following Google Drive link:

[Download trained model weights](https://drive.google.com/drive/folders/1XYLaFE9N9poPbmpT1gaOhEPvMYWutgB4?usp=sharing)

Then place the downloaded model weights under the `trained_model/` directory:

```text
trained_model/
├── gemma_lora/
├── gemma_lora_prompt/
├── go_lora/
└── nt_lora/
```

Then run:

```bash
cd LLM_prediction
python validate_lora_model.py
```

<img width="1234" height="168" alt="image" src="https://github.com/user-attachments/assets/7b78a1a2-19e7-4f4d-b745-bc18b206dae5" />


Run any training script directly:

If your machine has only one GPU, please set the GPU device index to 0 before running the scripts：
```bash
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
```

```bash
python train_gemma_opt.py
python train_gemma_prompt_opt.py
python train_genomeocean_opt.py
python train_nt_opt.py
```


For example, `train_gemma_opt.py` trains Gemma for promoter strength classification. The other `train` scripts follow the same workflow.


# For generative task
In this section, we use diffusion model to generate promoters.

### Step 1: Enviroment
```
git clone https://github.com/LX2004/LLM-promoter.git

cd LLM-promoter

conda env create -f environment.yml

conda activate promoter
```
### Step 2: Train diffusion model
```
cd E_coli_promoter_generation_core_region/code

python train_ddpm.py
```
### Step 3: Generate promoters
```
python generate_promoter.py
```
The trained generative model is stored in the `E_coli_promoter_generation_core_region/model/` folder, while the generated promoter data is stored in the `E_coli_promoter_generation_core_region/sequence/` folder.

# Tips
To facilitate researchers in utilizing the results of this study, we have stored the final synthesized promoter sequences in the Result folder.

| **File Name** | **Description** |
|--------------|---------------|
| Supplementary_file3_generation_promoter_by_VAE.fasta | Synthetic promoter sequences based on VAE |
| Supplementary_file4_generation_promoter_by_DDPM.fasta | Synthetic promoter sequences based on DDPM |
| Supplementary_file5_generation_promoter_by_DDPM_core_region.fasta | Promoters based on known core regions and DDPM |
| Supplementary_file6_generation_promoter_by_random_core_regions.fasta | Promoters composed of known core regions and randomly generated non-core regions |

