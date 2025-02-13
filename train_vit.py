import logging
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
import random
import ssl
import wandb
import hydra
import iecdt_lab
import torch
import torchvision

from torch import nn
from omegaconf import DictConfig, OmegaConf
from iecdt_lab.DiT import DiT
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import tqdm
import numpy as np

def setup_wandb(cfg):
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'project': cfg.project_name, 'config': config_dict, 'reinit': True, 'mode': cfg.wandb,
              'settings': wandb.Settings(_disable_stats=True)}
    run = wandb.init(**kwargs)
    #wandb.save('*.txt')
    #run.save()
    return cfg, run



def plot_reconstructions(batch, reconstructions, data_stats, max_images=8):
    fig, axes = plt.subplots(2, max_images, figsize=(15, 5))
    batch, reconstructions = batch[:max_images], reconstructions[:max_images]
    for i, (img, recon) in enumerate(zip(batch, reconstructions)):
        img = img.permute(1, 2, 0).cpu().numpy()
        img = img * data_stats["rgb_std"] + data_stats["rgb_mean"]
        recon = recon.permute(1, 2, 0).cpu().numpy()
        recon = recon * data_stats["rgb_std"] + data_stats["rgb_mean"]
        axes[0, i].imshow(img)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon)
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    fig.tight_layout()
    return fig


def validation(cfg, model, test_data_loader, data_stats):
    model.eval()
    running_mse = 0
    num_batches = len(test_data_loader)
    with torch.no_grad():
        for i, (batch, _) in enumerate(test_data_loader):
            batch = batch.to(cfg.device)
            reconstructions = model(batch)
            running_mse += torch.mean((batch - reconstructions) ** 2).cpu().numpy()

            if i == 0:
                fig = plot_reconstructions(batch, reconstructions, data_stats)

            if cfg.smoke_test and i == 10:
                num_batches = i + 1
                break

    val_mse = running_mse / num_batches
    return fig, val_mse


@hydra.main(version_base=None, config_path="config_vit", config_name="train")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    TILES_FILE = "/gws/nopw/j04/iecdt/deep_learning_lab/1km_naturalcolor_numpy"
    TRAIN_METADATA = "/gws/nopw/j04/iecdt/deep_learning_lab/1km_naturalcolor_metadata_time_train.csv"
    VAL_METADATA = "/gws/nopw/j04/iecdt/deep_learning_lab/1km_naturalcolor_metadata_time_val.csv"
    TEST_METADATA = "/gws/nopw/j04/iecdt/deep_learning_lab/1km_naturalcolor_metadata_time_test.csv"
    TILES_STATISTICS = "/gws/nopw/j04/iecdt/deep_learning_lab/1km_naturalcolor_metadata_rgb_stats.npz"
    
    setup_wandb(cfg)
    
    data_stats = np.load(TILES_STATISTICS)
    data_transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Grayscale(num_output_channels=1),
    torchvision.transforms.Normalize(
        mean=[float(data_stats["rgb_mean"][0])], 
        std=[float(data_stats["rgb_std"][0])]
    )
])
    
    train_data_loader, val_data_loader = iecdt_lab.data_loader.get_data_loaders(
        tiles_path=cfg.tiles_path,
        train_metadata=cfg.train_metadata,
        val_metadata=cfg.val_metadata,
        batch_size=cfg.batch_size,
        data_transforms=data_transforms,
        dataloader_workers=cfg.dataloader_workers,
    )

    model = DiT(input_size=256, in_channels=1, hidden_size=128, depth=8, num_heads=8, num_classes=0, learn_sigma= False,
                patch_size=8
                )
    model = model.to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        for i, (batch, _) in enumerate(train_data_loader):
            optimizer.zero_grad()

            batch = batch.to(cfg.device)
            preds = model(batch)
            loss = criterion(preds, batch)
            loss.backward()
            optimizer.step()
            wandb.log({"loss/train": loss.item()})

            # if i % cfg.log_freq == 0:
            #     logging.info(
            #         f"Epoch {epoch}/{cfg.epochs} Batch {i}/{len(train_data_loader)}: Loss={loss.item():.2f}"
            #     )

            # if cfg.smoke_test and i == 50:
            #     break

        eval_fig, val_mse = validation(cfg, model, val_data_loader, data_stats)
        wandb.log({"predictions": eval_fig, "loss/val": val_mse})

        if cfg.smoke_test:
            break

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()