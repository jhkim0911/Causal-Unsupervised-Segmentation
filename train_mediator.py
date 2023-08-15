import argparse

from tqdm import tqdm
from utils.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from loader.stego_dataloader import stego_dataloader
from torch.cuda.amp import autocast, GradScaler
from loader.netloader import network_loader, segment_loader

cudnn.benchmark = True
scaler = GradScaler()

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def ddp_clean():
    dist.destroy_process_group()

@Wrapper.EpochPrint
def train(args, net, segment, train_loader, optimizer):
    segment.eval()

    # patch size
    patch_size = int(args.ckpt.split('/')[-1].split('.')[0].split('_')[-1])

    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        img = batch["img"].cuda()

        # intermediate feature
        with autocast():
            feat = net(img)[:, 1:, :]

            # computing modularity based codebook
            loss_mod = segment.compute_modularity_based_codebook(segment.codebook, feat, grid=args.grid)

        # optimization
        optimizer.zero_grad()
        scaler.scale(loss_mod).backward()
        scaler.step(optimizer)
        scaler.update()

        # real-time print
        desc = f'[Train]'
        prog_bar.set_description(desc, refresh=True)

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

@Wrapper.TestPrint
def test(args, net, segment, nice, test_loader):
    segment.eval()

    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        img = batch["img"].cuda()
        label = batch["label"].cuda()

        # intermediate feature
        with autocast():
            feat = net(img)[:, 1:, :]

            # interp feat
            interp_dec_feat = F.interpolate(segment.transform(feat), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster preds
            cluster_preds = segment.quantize_index(segment.untransform(interp_dec_feat), segment.codebook)

        # nice evaluation
        _, desc_nice = nice.eval(cluster_preds, label)

        # real-time print
        desc = f'[TEST] {desc_nice}'
        prog_bar.set_description(desc, refresh=True)

    nice.reset()

    # Interrupt for sync GPU Process
    if args.distributed: dist.barrier()

def pickle_path_and_exist(args):
    from os.path import exists
    baseline = args.ckpt.split('/')[-1].split('.')[0]
    check_dir(f'CUSS/{args.dataset}/modularity/{baseline}/{args.num_codebook}')
    filepath = f'CUSS/{args.dataset}/modularity/{baseline}/{args.num_codebook}/modular.npy'
    return filepath, exists(filepath)

def main(rank, args, ngpus_per_node):
    # setup ddp process
    if args.distributed: ddp_setup(args, rank, ngpus_per_node)

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # print argparse
    print_argparse(args, rank)

    # dataset loader
    train_loader, test_loader, sampler = stego_dataloader(args)

    # network loader
    net = network_loader(args, rank)
    segment = segment_loader(args, rank)

    # optimizer and scheduler
    optimizer = torch.optim.AdamW(segment.parameters(), lr=1e-4 * ngpus_per_node)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.2)

    # evaluation
    nice = NiceTool(args.n_classes)

    ###################################################################################
    # train only modularity?
    path, is_exist = pickle_path_and_exist(args)

    # early save for time
    if not is_exist:
        rprint("No File Exists!!", rank)
        # train
        for epoch in range(args.epoch):

            # for shuffle
            if args.distributed: sampler.set_epoch(epoch)

            # train
            train(
                epoch,  # for decorator
                rank,  # for decorator
                args,
                net.module if args.distributed else net,
                segment.module if args.distributed else segment,
                train_loader,
                optimizer)

            test(
                epoch, # for decorator
                rank, # for decorator
                args,
                net.module if args.distributed else net,
                segment.module if args.distributed else segment,
                nice,
                test_loader)

            # scheduler step
            scheduler.step()

            # save
            if rank == 0:
                np.save(path, segment.module.codebook.detach().cpu().numpy()
                if args.distributed else segment.codebook.detach().cpu().numpy())

            # Interrupt for sync GPU Process
            if args.distributed: dist.barrier()

    else:
        rprint("Already Exists!!", rank)
    ###################################################################################


    # clean ddp process
    if args.distributed: ddp_clean()


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()

    # fixed parameter
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--distributed', default=True, type=str2bool)
    parser.add_argument('--load_Best', default=False, type=str2bool)
    parser.add_argument('--load_Fine', default=False, type=str2bool)
    parser.add_argument('--train_resolution', default=224, type=int)
    parser.add_argument('--test_resolution', default=320, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)

    # dataset and baseline
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='cocostuff27', type=str)
    parser.add_argument('--ckpt', default='checkpoint/dinov2_vit_base_14.pth', type=str)

    # DDP
    parser.add_argument('--gpu', default='4,5,6,7', type=str)
    parser.add_argument('--port', default='12359', type=str)

    # parameter
    parser.add_argument('--grid', default='yes', type=str2bool)
    parser.add_argument('--num_codebook', default=2048, type=int)

    args = parser.parse_args()

    if 'dinov2' in args.ckpt: args.test_resolution=322

    if 'small' in args.ckpt:
        args.dim=384
        args.reduced_dim=90
        args.projection_dim=2048
    elif 'base' in args.ckpt:
        args.dim=768
        args.reduced_dim=90
        args.projection_dim=2048

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    if args.distributed:
        # cuda visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        # multiprocess spawn
        mp.spawn(main, args=(args, ngpus_per_node), nprocs=ngpus_per_node, join=True)
    else:
        # first gpu index is activated once there are several gpu in args.gpu
        main(rank=gpu_list[0], args=args, ngpus_per_node=1)
