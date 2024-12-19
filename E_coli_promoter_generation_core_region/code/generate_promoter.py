import argparse
import torch
import torchvision
from utils import *
import script_utils
import os
import pdb
import pdb
from Bio import SeqIO


def get_promoter_by_fasta_file(file_name):
    
    sequences = []
    with open(file_name, 'r') as fasta_file:

        for record in SeqIO.parse(fasta_file, 'fasta'):

            sequences.append(str(record.seq))
    
    return sequences

def backbone_one_hot(seq):
    charmap = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            encoded[:, i] = np.random.rand(4)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded


def main():

    args = create_argparser().parse_args()

    nat = get_promoter_by_fasta_file(file_name='../data/weak_strong_promoter.fasta')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)

    encoded_sequences = []
    for sequence in nat:

        if len(sequence) != args.promoter_length:
            print('error!!!')

        encoded_sequence = backbone_one_hot(sequence)
        encoded_sequences.append(encoded_sequence)
        
    encoded_arrary = np.array(encoded_sequences)
    encoded_arrary = np.expand_dims(encoded_arrary, axis=1)

    try:
        for epoch in range(50,500,50):

            diffusion = script_utils.get_diffusion_from_args(args).to(device)
           
            model_path = f'../model/ecoli-ddpm-2024-09-22-13-55-iteration-{epoch}--kernel=7--GPU_ID=7--model.pth'
           
            print(' model_path', model_path)

            diffusion.load_state_dict(torch.load(model_path))
            sequences = []         

            ori_x = torch.tensor(encoded_arrary, dtype=torch.float32).to(device)
            samples = diffusion.sample_from_partial_noise(ori_x, device)
            samples = samples.squeeze(dim=1)
            samples = samples.to('cpu').detach().numpy()
            sequences = []

            for i in range(samples.shape[0]):

                decoded_sequence = decode_one_hot(samples[i])
                sequences.append(decoded_sequence)

            k_mer_fre_cor = calculate_overall_kmer_correlation(dataset1=sequences, dataset2=nat, k=6)
            print('6_mer_fre_cor = ', k_mer_fre_cor)
            make_fasta_file(sequences,path=f'../sequence/ecoli-ddpm-2024-09-22-13-55-iteration-{epoch}--kernel=7--GPU_ID=7-_6_mer_fre_cor={k_mer_fre_cor}.fasta') 


    except KeyboardInterrupt:
        print("Keyboard interrupt, generation finished early")


def create_argparser():

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    defaults = dict(num_images=1024, device=device, schedule_low=1e-4,
    schedule_high=0.02,out_init_conv_padding = 3, promoter_length = 81)
    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--save_dir", type=str)
    script_utils.add_dict_to_argparser(parser, defaults)

    return parser

if __name__ == "__main__":
    main()