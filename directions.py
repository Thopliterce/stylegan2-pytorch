import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import os
from model import Generator, Discriminator
from PIL import Image
import math
import time

def main():
    device = "cuda"

    parser = argparse.ArgumentParser()

    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--genmodel', type=str, default="checkpoint/network-snapshot-017325.pt")
    parser.add_argument("--ckpt", type=str, default="checkpoint/direction.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--round', type=int, default=100000)
    parser.add_argument('--norm', type=float, default=1.0)

    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8

    ckpt = torch.load(args.genmodel)
    gen = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    gen.load_state_dict(ckpt['g_ema'])
    norm = args.norm

    # Abusing the last class as the distance value
    classifier = Discriminator(size=args.size, inputs=6, outputs=args.dim*2).to(device)
    dirs = torch.randn(args.dim, args.latent, device=device, requires_grad=True)
    loss1 = nn.CrossEntropyLoss()
    loss2 = nn.MSELoss(reduction='sum')
    opt = optim.Adam([
        {'params': classifier.parameters(), 'lr': 1e-3},
        {'params': [dirs], 'lr': 1e-2}
    ])
    dist_stddev = math.sqrt(args.latent)
    if os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt)
        classifier.load_state_dict(ckpt['classifier'])
        dirs = ckpt['dirs'].to(device)
        opt.load_state_dict(ckpt['opt'])

    correct, total = 0, 0
    step, last_time = 0, time.time()
    for i in range(args.round):
        step += 1
        # Prepare random latent vectors and indices and dists without gradient
        with torch.no_grad():
            dirs = F.normalize(dirs).detach()
            latents = torch.randn(args.batch, args.latent, device=device)
            latents = gen.get_latent(latents)
            # Currently we use 512-dim latents rather than 18,512-dim latents
            dir_dists = torch.randn(args.batch, device=device)*dist_stddev
            dir_indices = torch.randint(low=0, high=args.dim, size=(args.batch,), device=device)
            dists = torch.zeros(args.batch, args.dim, device=device)
            dists[torch.arange(args.batch), dir_indices] = dir_dists

        opt.zero_grad()
        # Compute the actual latent translation vector, which need to be differentiable
        dir_vector = torch.index_select(dirs, 0, dir_indices) * dir_dists[:, None] * norm

        latents1 = latents - dir_vector * 0.5
        latents2 = latents + dir_vector * 0.5
        images1, _ = gen([latents1], input_is_latent=True)
        images2, _ = gen([latents2], input_is_latent=True)
        full_images = torch.cat([images1, images2], dim=1)
        predictions = classifier(full_images)
        # Compute accuracy as well
        _, predicted_indices = torch.max(predictions[:, :args.dim], 1)
        total += args.batch
        correct += (predicted_indices == dir_indices).sum().item()

        # The first `dim` values are the classification results, and
        # the last `dim` values are the expected distance
        loss1val = loss1(predictions[:, :args.dim], dir_indices)
        loss2val = loss2(predictions[:, args.dim:args.dim*2], dists) / args.latent
        loss = loss1val + loss2val
        loss.backward()
        opt.step()
        if i % 10 == 9:
            print("round %d done with loss1=%f loss2=%f acc=%d/%d time=%f" %
                (i+1, loss1val.item(), loss2val.item(), correct, total, (time.time()-last_time)/step))
            step, last_time = 0, time.time()
            correct, total = 0, 0
            # TODO: better rename
            tmp_file = args.ckpt + '.tmp'
            torch.save({
                'classifier': classifier.state_dict(),
                'dirs': dirs,
                'opt': opt.state_dict()
            }, tmp_file)
            os.rename(tmp_file, args.ckpt)
            img = torch.cat([images1, images2]).reshape((2,) + images1.shape)
            img = img.permute((2, 0, 3, 1, 4))
            img = img.reshape((img.shape[0], img.shape[1]*img.shape[2], img.shape[3]*img.shape[4]))
            torchvision.utils.save_image([img], "sample/%d.png" % i, normalize=True, range=(-1, 1))

def show_directions():
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=64)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--genmodel', type=str, default="checkpoint/network-snapshot-017325.pt")
    parser.add_argument("--ckpt", type=str, default="checkpoint/direction.pt")
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument("--step", type=float, default=0.3)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--sample", type=int, default=3)

    args = parser.parse_args()
    args.latent = 512
    args.n_mlp = 8
    dist_stddev = math.sqrt(args.latent)

    ckpt = torch.load(args.genmodel)
    gen = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    gen.load_state_dict(ckpt['g_ema'])

    ckpt = torch.load(args.ckpt)
    dirs = ckpt['dirs'].to(device)
    assert(dirs.shape[0] == args.dim)
    row_size = args.count*2+1

    with torch.no_grad():
        center_latent = gen.get_latent(torch.randn(args.sample, args.latent, device=device))
        for i in range(args.dim):
            steps = torch.arange(-args.step*args.count, args.step*(args.count+0.01), args.step, device=device)*dist_stddev
            latents = center_latent[:, None, :] + steps[None, :, None] * dirs[i, None, None, :]
            latents_plain = latents.reshape((args.sample*row_size, args.latent))
            images = torch.empty((args.sample*row_size, 3, args.size, args.size), device=device)
            for j in range(len(latents_plain)):
                cur_image, _ = gen([latents_plain[j:j+1]], input_is_latent=True)
                images[j] = cur_image[0]
            torchvision.utils.save_image(images, "sample/dir-%d.png" % i, nrow=row_size, normalize=True, range=(-1, 1))

# show_directions()
main()
