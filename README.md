# Paper

Code used in the paper: A Hybrid Framework Combining Large Language Model-Based Core Motif Identification and Diffusion Model-Driven Adjacent Sequence Generation for High-Performance Promoter Design.

# Framework

![image](https://github.com/user-attachments/assets/e834b75f-635b-4943-bc2c-cd1990e59ace)

### Strength Prediction  
The LLM (Gemma2) predicts the strength (strong or weak) of an input sequence provided by the user. If the prediction is incorrect, the model is fine-tuned using the LoRA method to improve accuracy.  

### Pseudo-sequence Mutation  
The input sequence undergoes mutations to generate multiple variants, enabling the analysis of different promoter regions' contributions to strength.  

### Core Region Identification  
By evaluating the impact of mutations on predicted strength, the core region of the sequence—where variations significantly affect strength—is identified.  

### Promoter Synthesis  
Based on the identified core region, a diffusion model reconstructs non-core regions, ultimately generating a complete promoter sequence.  
