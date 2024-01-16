import os
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset
from model import build_model


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("model", choices=["Transformer", "LSTM", "MyModel"])
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--in-size", type=int, default=96)
    parser.add_argument("--out_size", type=int, default=96, choices=[96, 336])
    parser.add_argument("--epochs", type=int, default=500)
    args = parser.parse_args()

    test_dataset = MyDataset(f"{args.dataset}.csv",split='test',
                              in_size=args.in_size, out_size=args.out_size,
                              seed=args.seed)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    model = build_model(args.model, args.out_size)
    log_dir = "./logs/{}/{}".format('/data/ETTh1.csv', args.model)
    checkpoint = torch.load(os.path.join(log_dir, "checkpoint_best_336.pth"), map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)

    min_mae = float("inf")
    to_plot = {"inputs": None, "outputs": None, "targets": None}
    mae_list = []
    for inputs, targets in tqdm(test_loader):
        inputs = inputs.to(args.device)
        targets = targets.to(args.device)
        with torch.no_grad():
            outputs = model(inputs)
        loss = F.l1_loss(outputs[..., -1], targets[..., -1], reduction="none")
        loss = loss.mean(dim=-1)
        mae_list.extend(loss.tolist())
        ind = loss.argmin()
        mae = loss[ind]
        if mae < min_mae:
            min_mae = mae
            to_plot.update({
                "inputs": inputs[ind, :, -1].cpu().numpy(),
                "outputs": outputs[ind, :, -1].cpu().numpy(),
                "targets": targets[ind, :, -1].cpu().numpy(),
            })
    mae_list = np.array(mae_list)
    mean_mae = np.mean(mae_list)
    print("MAE = {:.4f}".format(mean_mae))

    inputs = to_plot["inputs"]
    outputs = to_plot["outputs"]
    targets = to_plot["targets"]
    groundtruth = np.concatenate((inputs, targets))
    prediction = np.concatenate((inputs, outputs))
    plt.plot(range(len(prediction)), prediction, label="Prediction")
    plt.plot(range(len(groundtruth)), groundtruth, label="GroundTruth")
    plt.legend()
    plt.savefig(f"prediction_lstm_{args.out_size}.png")
