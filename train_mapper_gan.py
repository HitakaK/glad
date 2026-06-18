import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from lowdim_mapper import LowDimMapper
from glad_utils import load_sgxl


class Discriminator(nn.Module):
    def __init__(self, img_channels=3, base=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, base, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base, base * 2, 4, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base * 2, base * 4, 4, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Flatten(),
            nn.Linear(base * 4 * 4 * 4, 1),
        )

    def forward(self, x):
        return self.net(x).view(-1)


def generate_fake(G, mapper, z, sg_batch):
    outs = []
    for z_split in torch.split(z, sg_batch):
        wplus = mapper(z_split)
        fake_split = G(wplus, mode="wp")
        outs.append(fake_split)
    return torch.cat(outs, dim=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="CIFAR10")
    p.add_argument("--data_path", default="./data")
    p.add_argument("--save_path", default="./mapper_ckpt")
    p.add_argument("--u_dim", type=int, default=4)
    p.add_argument("--mapper_hidden", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--lr_g", type=float, default=1e-4)
    p.add_argument("--lr_d", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--sg_batch", type=int, default=64)
    p.add_argument("--device", default="cuda")

    p.add_argument("--special_gan", default=None)
    p.add_argument("--rand_gan_un", action="store_true")
    p.add_argument("--rand_gan_con", action="store_true")
    
    args = p.parse_args()

    os.makedirs(args.save_path, exist_ok=True)
    device = args.device if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path,
        train=True,
        download=True,
        transform=transform,
    )

    loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
    )

    # G: fixed StyleGAN
    G, zdim, w_dim, num_ws = load_sgxl(32, args)
    G.eval()
    for p_ in G.parameters():
        p_.requires_grad_(False)

    mapper = LowDimMapper(
        u_dim=args.u_dim,
        num_ws=num_ws,
        w_dim=w_dim,
        hidden=args.mapper_hidden,
    ).to(device)

    D = Discriminator().to(device)

    opt_g = optim.Adam(mapper.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    opt_d = optim.Adam(D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

    loss_fn = nn.BCEWithLogitsLoss()

    data_iter = iter(loader)

    for step in range(1, args.steps + 1):
        try:
            real, _ = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            real, _ = next(data_iter)

        real = real.to(device)
        bsz = real.shape[0]

        # -----------------
        # Train D
        # -----------------
        z = torch.randn(bsz, args.u_dim, device=device)

        with torch.no_grad():
            fake = generate_fake(G, mapper, z, args.sg_batch)

        real_logits = D(real)
        fake_logits = D(fake.detach())

        d_loss = (
            loss_fn(real_logits, torch.ones_like(real_logits)) +
            loss_fn(fake_logits, torch.zeros_like(fake_logits))
        ) * 0.5

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # -----------------
        # Train Mapper
        # -----------------
        z = torch.randn(bsz, args.u_dim, device=device)
        fake = generate_fake(G, mapper, z, args.sg_batch)

        fake_logits = D(fake)
        g_loss = loss_fn(fake_logits, torch.ones_like(fake_logits))

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        if step % 100 == 0:
            print(f"step={step} d_loss={d_loss.item():.4f} g_loss={g_loss.item():.4f}")

        if step % 1000 == 0 or step == args.steps:
            save_file = os.path.join(args.save_path, f"mapper_u{args.u_dim}_step{step}.pt")
            torch.save({
                "model_state_dict": mapper.state_dict(),
                "u_dim": args.u_dim,
                "num_ws": 12,
                "w_dim": 512,
                "hidden": args.mapper_hidden,
                "step": step,
            }, save_file)
            print(f"saved: {save_file}")


if __name__ == "__main__":
    main()
