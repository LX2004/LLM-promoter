# Paper

Code used in the paper: Decoding and Rewiring Promoter Architecture Using Large Language Models and Diffusion Frameworks.

# Framework

![image](https://github.com/user-attachments/assets/2c176dea-8d84-423a-b418-a1f4e221c772)

### Strength Prediction  
The LLM (Gemma2) predicts the strength (strong or weak) of an input sequence provided by the user. If the prediction is incorrect, the model is fine-tuned using the LoRA method to improve accuracy.  

### Pseudo-sequence Mutation  
The input sequence undergoes mutations to generate multiple variants, enabling the analysis of different promoter regions' contributions to strength.  

### Core Region Identification  
By evaluating the impact of mutations on predicted strength, the core region of the sequence—where variations significantly affect strength—is identified.  

### Promoter Synthesis  
Based on the identified core region, a diffusion model reconstructs non-core regions, ultimately generating a complete promoter sequence.  

# For classification task
For LLM environment configuration, please refer to:
- [Self-LLM Gemma2 LoRA Fine-tuning Guide](https://github.com/datawhalechina/self-llm/blob/master/models/Gemma2/04-Gemma-2-9b-it%20peft%20lora%E5%BE%AE%E8%B0%83.md)  
- [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory/tree/main)

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
conda activate Gemma2
```
After setting up the virtual environment, the user can run the `AMP_classification_Gemma2/code/fine_tune_classify_AMP_peptide.py` script to fine-tune the large language model.

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

