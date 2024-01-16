from argparse import ArgumentParser
from dataset import build_data
import torch
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from model import ETTLSTM, ETTTransformer


def criterion(input: torch.Tensor, target: torch.Tensor):
    return F.mse_loss(input, target)
    # return F.l1_loss(input, target)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("model", choices=["lstm", "transformer"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred-len", type=int, default=96, choices=[96, 336])
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    train_data = build_data(f"data/{args.dataset}.csv", "train", args.seq_len, args.pred_len)
    train_dataset = TensorDataset(*map(torch.from_numpy, train_data))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_data = build_data(f"data/{args.dataset}.csv", "val", args.seq_len, args.pred_len)
    val_dataset = TensorDataset(*map(torch.from_numpy, val_data))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    if args.model == "lstm":
        model = ETTLSTM()
    elif args.model == "transformer":
        model = ETTTransformer()
    else:
        raise NotImplementedError()
    model.to(args.device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = LambdaLR(optimizer, lambda epoch: 0.5 ** epoch)

    min_mae = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            # prd = model(src, torch.cat((src, torch.zeros_like(tgt)), dim=1))
            # prd = prd[:, src.size(1):]
            prd = model(src, torch.zeros_like(tgt))
            loss = criterion(prd, tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        model.eval()
        mae_list = []
        for src, tgt in tqdm(val_loader, desc="val", leave=False):
            src = src.to(args.device)
            tgt = tgt.to(args.device)
            with torch.no_grad():
                # prd = model(src, torch.cat((src, torch.zeros_like(tgt)), dim=1))
                # prd = prd[:, src.size(1):]
                prd = model(src, torch.zeros_like(tgt))
            mae_list.extend(
                F.l1_loss(prd[..., -1], tgt[..., -1], reduction="none").tolist())
        mae = torch.tensor(mae_list).mean()
        if mae < min_mae:
            print(f"Epoch {epoch:3d}: MAE = {mae:.4f}")
            min_mae = mae
            checkpoint = {"model": model.state_dict()}
            torch.save(checkpoint, f"checkpoint_best_{args.model}_{args.pred_len}.pth")
