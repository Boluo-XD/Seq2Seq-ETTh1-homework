import os
from argparse import ArgumentParser
import numpy as np
from dataset import MyDataset
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm


from model import build_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", default='./data/ETTh1.csv',choices=["ETTh1", "ETTh2"])
    parser.add_argument("model", choices=["Transformer", "LSTM", "MyModel"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--in-size", type=int, default=96)
    parser.add_argument("--out_size", type=int, default=96, choices=[96, 336])
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    train_dataset = MyDataset(args.dataset,split='train',
                              in_size=args.in_size, out_size=args.out_size,
                              seed=args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = MyDataset(args.dataset,split='val',
                            in_size=args.in_size, out_size=args.out_size,
                            seed=args.seed)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = build_model(args.model, args.out_size)
    model.to(args.device)
    optimizer = AdamW(model.parameters(), lr=1e-3)

    min_mae = float("inf")
    log_dir = "./logs/{}/{}".format(args.dataset, args.model)
    logger = SummaryWriter(log_dir=log_dir)
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch:5d}", leave=False):
            optimizer.zero_grad()
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            outputs = model(inputs, targets)
            loss = F.l1_loss(outputs, targets)
            loss.backward()
            optimizer.step()
            logger.add_scalar("loss", loss, global_step)
            global_step += 1
        model.eval()
        losses = []
        for inputs, targets in val_loader:
            inputs = inputs.to(args.device)
            targets = targets.to(args.device)
            with torch.no_grad():
                outputs = model(inputs)
            loss = F.l1_loss(outputs[..., -1], targets[..., -1], reduction="none")
            losses.extend(loss.tolist())
        mae = np.mean(np.array(losses))
        if mae < min_mae:
            print("Epoch {:5d}: save checkpoint, with MAE = {:.4f}.".format(epoch, mae))
            min_mae = mae
            checkpoint = {"model": model.state_dict(), "epoch": epoch}
            torch.save(checkpoint, os.path.join(log_dir, f"checkpoint_best_{args.out_size}.pth"))
        if epoch%10 == 0 and mae<0.05:
            torch.save(checkpoint,os.path.join(log_dir, f"checkpoint_best_{args.out_size}_{epoch}_{mae}.pth"))