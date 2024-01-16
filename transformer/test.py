from argparse import ArgumentParser
from dataset import build_data
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


from model import ETTTransformer,ETTLSTM


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("dataset", choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2"])
    parser.add_argument("model", choices=["transformer","lstm"])
    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96, choices=[96, 336])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    test_data = build_data(f"data/{args.dataset}.csv", "train", args.seq_len, args.pred_len)
    test_dataset = TensorDataset(*map(torch.from_numpy, test_data))
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    if args.model == "transformer":
        model = ETTTransformer()
    elif args.model == "lstm":
        model = ETTLSTM()
    else:
        raise NotImplementedError()
    checkpoint = torch.load("checkpoint_best_transformer.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model.to(args.device)

    min_mse = float("inf")
    to_plot = {"src": None, "prd": None, "tgt": None}
    mse_list = []
    for src, tgt in tqdm(test_loader):
        src = src.to(args.device)
        tgt = tgt.to(args.device)
        with torch.no_grad():
            prd = model(src, torch.cat((src, torch.zeros_like(tgt)), dim=1))
            prd = prd[:, src.size(1):]
        loss = F.mse_loss(prd[..., -1], tgt[..., -1], reduction="none").mean(dim=-1)
        mse_list.extend(loss.tolist())
        ind = loss.argmin()
        mse = loss[ind]
        if mse < min_mse:
            min_mse = mse
            to_plot.update({
                "src": src[ind, :, -1].cpu().numpy(),
                "prd": prd[ind, :, -1].cpu().numpy(),
                "tgt": tgt[ind, :, -1].cpu().numpy()
            })
    mse_list = np.array(mse_list)
    mean_mse = np.mean(mse_list)
    print("MSE = {:.4f}".format(mean_mse))

    src = to_plot["src"]
    prd = to_plot["prd"]
    tgt = to_plot["tgt"]
    pred = np.concatenate((src, prd))
    gt = np.concatenate((src, tgt))
    plt.plot(range(len(pred)), pred, label="Prediction")
    plt.plot(range(len(gt)), gt, label="GroundTruth")
    plt.legend()
    plt.savefig(f"result_{args.model}_{args.pred_len}.png")
