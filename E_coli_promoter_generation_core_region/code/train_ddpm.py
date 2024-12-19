import argparse
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader,Dataset
# from torchvision import datasets
import script_utils
from torch.optim.lr_scheduler import StepLR
from utils import *
import pdb
from Bio import SeqIO

def get_promoter_by_fasta_file(file_name):
    sequences = []

    with open(file_name, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):

            sequences.append(str(record.seq))
    
    print(sequences[0])
    print('number：', len(sequences))

    return sequences

class CustomDataset(Dataset):
    def __init__(self, data_folder):
        self.data = data_folder

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return torch.tensor(sample, dtype=torch.float32)

def main():

    loss_flag = 0.15 
    args = create_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID = 0
    torch.cuda.set_device(GPU_ID)

    try:

        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))

        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        batch_size = args.batch_size

        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)
        all_sample = get_promoter_by_fasta_file(file_name='../data/weak_strong_promoter.fasta')
        print(all_sample[0])
        all_sample_number = len(all_sample)


        train_size = int(0.8 *  all_sample_number)  
        encoded_sequence_train = []

        for sequence in all_sample[:train_size]:

            if len(sequence) != args.promoter_length:
                print('error!!!')

            encoded_sequence = one_hot_encoding(sequence)
            encoded_sequence_train.append(encoded_sequence)
            
        encoded_sequence_test = []
        for sequence in all_sample[train_size:]:

            if len(sequence) != args.promoter_length:
                print('error!!!')
            
            encoded_sequence = one_hot_encoding(sequence)
            encoded_sequence_test.append(encoded_sequence)

        train_arrary = np.array(encoded_sequence_train)
        test_arrary = np.array(encoded_sequence_test)

        print('train_arrary.shape = ', train_arrary.shape)
        print('test_arrary.shape = ', test_arrary.shape)

        train_arrary = np.expand_dims(train_arrary, axis=1)
        test_arrary = np.expand_dims(test_arrary, axis=1)


        train_dataset = CustomDataset(train_arrary)
        test_dataset = CustomDataset(test_arrary)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        acc_train_loss = 0

        for iteration in range(1, args.iterations + 1):

            diffusion.train()

            for x in train_loader:
                # x, y = next(train_loader)
                x = x.to(device)
                # y = y.to(device)
                
                loss = diffusion(x)

                acc_train_loss += loss.item()
                # train_loss_epoch += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                diffusion.update_ema()

            print(f'epoch ={iteration}, train loss = {acc_train_loss}')


            scheduler.step()
            
            if iteration % args.log_rate == 0:
                test_loss = 0

                with torch.no_grad():

                    diffusion.eval()
                    
                    for x in test_loader:

                        x = x.to(device)

                        
                        loss = diffusion(x)

                        test_loss += loss.item()
                

                test_loss /= len(test_loader)
                acc_train_loss /= args.log_rate
            
                print(f'epoch = {iteration}, test_loss = {test_loss}')
            
            acc_train_loss = 0

            if test_loss < loss_flag:

                loss_flag = test_loss
                print('save best model')

                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-kernel={1+2*args.out_init_conv_padding}--GPU_ID={GPU_ID}--best-model.pth"
                torch.save(diffusion.state_dict(), model_filename)

            if iteration % args.checkpoint_rate == 0:

                model_filename = f"{args.log_dir}/{args.project_name}-{args.run_name}-iteration-{iteration}--kernel={1+2*args.out_init_conv_padding}--GPU_ID={GPU_ID}--model.pth"
                torch.save(diffusion.state_dict(), model_filename)
               
    except KeyboardInterrupt:

        print("Keyboard interrupt, run finished early")

def create_argparser():

    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")

    defaults = dict(

        learning_rate=1e-3,
        batch_size=512,
        iterations=500,

        log_to_wandb=True,
        log_rate=1,
        checkpoint_rate=50,
        log_dir="../model",
        project_name='ecoli',
        out_init_conv_padding = 3,
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,
        promoter_length = 81,

        start_1 = 25,
        end_1 = 35,

        start_2 = 48,
        end_2 = 56,

        start_3 = 59,
        end_3 = 62,

    )

    defaults.update(script_utils.diffusion_defaults())

    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()