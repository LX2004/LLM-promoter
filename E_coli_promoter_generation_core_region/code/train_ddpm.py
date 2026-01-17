import argparse
import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
import script_utils
from torch.optim.lr_scheduler import StepLR
from utils import *
from Bio import SeqIO
import os

# ===================== Data utils =====================

def get_promoter_by_fasta_file(file_name):
    sequences = []
    with open(file_name, 'r') as fasta_file:
        for record in SeqIO.parse(fasta_file, 'fasta'):
            sequences.append(str(record.seq))
    print("Example sequence:", sequences[0])
    print("Number of sequences:", len(sequences))
    return sequences


class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx], dtype=torch.float32)


# ===================== Main =====================

def main():

    args = create_argparser().parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID = 0
    torch.cuda.set_device(GPU_ID)

    # ---------- Early stopping parameters ----------
    EARLY_STOP_WINDOW = 10          # consecutive epochs
    EARLY_STOP_DELTA = 0.05         # 5% relative change
    val_loss_history = []

    try:
        diffusion = script_utils.get_diffusion_from_args(args).to(device)
        optimizer = torch.optim.Adam(diffusion.parameters(), lr=args.learning_rate)

        if args.model_checkpoint is not None:
            diffusion.load_state_dict(torch.load(args.model_checkpoint))

        if args.optim_checkpoint is not None:
            optimizer.load_state_dict(torch.load(args.optim_checkpoint))

        scheduler = StepLR(optimizer, step_size=50, gamma=0.5)

        # ---------- Load data ----------
        all_sample = get_promoter_by_fasta_file(
            file_name='../data/weak_strong_promoter.fasta'
        )
        n_total = len(all_sample)
        train_size = int(0.8 * n_total)

        encoded_train, encoded_test = [], []

        for seq in all_sample[:train_size]:
            encoded_train.append(one_hot_encoding(seq))

        for seq in all_sample[train_size:]:
            encoded_test.append(one_hot_encoding(seq))

        train_array = np.expand_dims(np.array(encoded_train), axis=1)
        test_array  = np.expand_dims(np.array(encoded_test), axis=1)

        train_loader = DataLoader(
            CustomDataset(train_array),
            batch_size=args.batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            CustomDataset(test_array),
            batch_size=args.batch_size,
            shuffle=False
        )

        best_val_loss = float("inf")

        # ===================== Training loop =====================
        for epoch in range(1, args.iterations + 1):

            diffusion.train()
            train_loss_epoch = 0.0

            for x in train_loader:
                x = x.to(device)
                loss = diffusion(x)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                diffusion.update_ema()

                train_loss_epoch += loss.item()

            train_loss_epoch /= len(train_loader)
            scheduler.step()

            # ---------- Validation ----------
            diffusion.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x in test_loader:
                    x = x.to(device)
                    val_loss += diffusion(x).item()
            val_loss /= len(test_loader)

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss: {train_loss_epoch:.6f} | "
                f"Val loss: {val_loss:.6f}"
            )

            # ---------- Save best model ----------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                os.makedirs(args.log_dir, exist_ok=True)
                best_model_path = (
                    f"{args.log_dir}/"
                    f"{args.project_name}-{args.run_name}-best-model.pth"
                )
                torch.save(diffusion.state_dict(), best_model_path)
                print("✓ Best model updated")

            # ---------- Early stopping check ----------
            val_loss_history.append(val_loss)
            if len(val_loss_history) >= EARLY_STOP_WINDOW:
                recent_losses = val_loss_history[-EARLY_STOP_WINDOW:]
                max_loss = max(recent_losses)
                min_loss = min(recent_losses)

                if (max_loss - min_loss) / max_loss < EARLY_STOP_DELTA:
                    print(
                        f"Early stopping triggered at epoch {epoch}: "
                        f"validation loss change < {EARLY_STOP_DELTA*100:.1f}% "
                        f"over {EARLY_STOP_WINDOW} epochs."
                    )
                    break

            # ---------- Periodic checkpoint ----------
            if epoch % args.checkpoint_rate == 0:
                ckpt_path = (
                    f"{args.log_dir}/"
                    f"{args.project_name}-{args.run_name}-epoch-{epoch}.pth"
                )
                torch.save(diffusion.state_dict(), ckpt_path)

    except KeyboardInterrupt:
        print("Training interrupted manually.")


# ===================== Args =====================

def create_argparser():
    run_name = datetime.datetime.now().strftime("ddpm-%Y-%m-%d-%H-%M")

    defaults = dict(
        learning_rate=1e-3,
        batch_size=512,
        iterations=500,

        log_rate=1,
        checkpoint_rate=50,
        log_dir="../model",
        project_name='ecoli',
        out_init_conv_padding=3,
        run_name=run_name,

        model_checkpoint=None,
        optim_checkpoint=None,

        schedule_low=1e-4,
        schedule_high=0.02,
        promoter_length=81,

        # core regions
        start_1=24, end_1=28,
        start_2=29, end_2=32,
        start_3=46, end_3=56,
        start_4=58, end_4=61,
    )

    defaults.update(script_utils.diffusion_defaults())
    parser = argparse.ArgumentParser()
    script_utils.add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
