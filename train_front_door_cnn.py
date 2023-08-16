import argparse

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


from tqdm import tqdm
from utils.utils import *
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn
from loader.stego_dataloader import stego_dataloader
from torch.cuda.amp import autocast, GradScaler
from loader.netloader import network_loader, segment_cnn_loader
from tensorboardX import SummaryWriter

cudnn.benchmark = True
scaler = GradScaler()
scaler_cluster = GradScaler()

# tensorboard
counter = 0
counter_test = 0

def ddp_setup(args, rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = args.port

    # initialize
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def ddp_clean():
    dist.destroy_process_group()


@Wrapper.EpochPrint
def train(args, net, segment, train_loader, optimizer, writer, rank):
    global counter
    segment.train()
    segment.bank_init()

    total_acc = 0
    total_loss = 0
    total_loss_front = 0
    total_loss_linear = 0

    prog_bar = tqdm(enumerate(train_loader), total=len(train_loader), leave=True)
    for idx, batch in prog_bar:

        # optimizer
        with autocast():

            # image and label and self supervised feature
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            # intermediate features
            feat = net(img)[:, 1:, :]
            orig_seg_feat_ema = segment.head_ema(feat, drop=segment.dropout)

            if args.grid: feat, _ = segment.stochastic_sampling(feat)

            ######################################################################
            # teacher
            seg_feat_ema = segment.head_ema(feat, drop=segment.dropout)
            proj_feat_ema = segment.projection_head_ema(seg_feat_ema, drop=segment.dropout)
            ######################################################################

            ######################################################################
            # student
            seg_feat = segment.head(feat, drop=segment.dropout)
            proj_feat = segment.projection_head(seg_feat, drop=segment.dropout)
            ######################################################################

            ######################################################################
            # bank compute and contrastive loss
            segment.bank_compute()
            loss_front =  segment.contrastive_ema_with_codebook_bank(feat, proj_feat, proj_feat_ema)
            ######################################################################

            # linear probe loss
            linear_logits = segment.linear(orig_seg_feat_ema)
            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
            flat_linear_logits = linear_logits.permute(0, 2, 3, 1).view(-1, args.n_classes)
            flat_label = label.view(-1)
            flat_label_mask = (flat_label >= 0) & (flat_label < args.n_classes)
            loss_linear = F.cross_entropy(flat_linear_logits[flat_label_mask], flat_label[flat_label_mask])

            # cluster loss
            loss_cluster, _ = segment.forward_centroid(orig_seg_feat_ema)

            # loss
            loss = loss_front + loss_linear + loss_cluster

        # optimizer
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # ema update
        segment.ema_update(segment.head, segment.head_ema)
        segment.ema_update(segment.projection_head, segment.projection_head_ema)

        # bank update
        segment.bank_update(feat, proj_feat_ema)

        # linear probe acc check
        pred_label = linear_logits.argmax(dim=1)
        flat_pred_label = pred_label.view(-1)
        acc = (flat_pred_label[flat_label_mask] == flat_label[flat_label_mask]).sum() / flat_label[
            flat_label_mask].numel()
        total_acc += acc.item()

        # loss check
        total_loss += loss.item()
        total_loss_front += loss_front.item()
        total_loss_linear += loss_linear.item()

        # real-time print
        desc = f'[Train] Loss: {total_loss / (idx + 1):.2f}={total_loss_front / (idx + 1):.2f}+{total_loss_linear / (idx + 1):.2f}'
        desc += f' ACC: {100. * total_acc / (idx + 1):.1f}%'
        prog_bar.set_description(desc, refresh=True)


        # tensorboard
        if (args.distributed == True) and (rank == 0):
            writer.add_scalar('Train/Contrastive', loss_front, counter)
            writer.add_scalar('Train/Linear', loss_linear, counter)
            writer.add_scalar('Train/Cluster', loss_cluster, counter)
            writer.add_scalar('Train/Acc', total_acc / (idx + 1), counter)
            counter += 1

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()


@Wrapper.TestPrint
def test(args, net, segment, nice, test_loader, writer, rank):
    global counter_test
    segment.eval()

    total_acc = 0
    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        img = batch["img"].cuda()
        label = batch["label"].cuda()

        # intermediate feature
        with autocast():
            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat)

            # linear probe loss
            linear_logits = segment.linear(seg_feat_ema)
            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
            flat_label = label.view(-1)
            flat_label_mask = (flat_label >= 0) & (flat_label < args.n_classes)

            # interp feat
            interp_seg_feat = F.interpolate(segment.transform(seg_feat_ema), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster
            cluster_preds = segment.forward_centroid(segment.untransform(interp_seg_feat), inference=True)

        # linear probe acc check
        pred_label = linear_logits.argmax(dim=1)
        flat_pred_label = pred_label.view(-1)
        acc = (flat_pred_label[flat_label_mask] == flat_label[flat_label_mask]).sum() / flat_label[
            flat_label_mask].numel()
        total_acc += acc.item()

        # nice evaluation
        metric, desc_nice = nice.eval(cluster_preds, label)


        # tensorboard
        if (args.distributed == True) and (rank == 0):
            writer.add_scalar('Test/mIoU', metric["mIoU"], counter_test)
            writer.add_scalar('Test/mAP', metric["mAP"], counter_test)
            writer.add_scalar('Test/Acc', metric["Acc"], counter_test)
            writer.add_scalar('Test/Linear', 100. * total_acc / (idx + 1), counter_test)
            counter_test += 1

        # real-time print
        desc = f'[TEST] Acc (Linear): {100. * total_acc / (idx + 1):.1f}% | {desc_nice}'
        prog_bar.set_description(desc, refresh=True)

    # evaluation metric reset
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
    segment = segment_cnn_loader(args, rank)

    # Bank and EMA initialization
    if args.distributed:
        segment.module.bank_init()
        segment.module.ema_init(segment.module.head, segment.module.head_ema)
        segment.module.ema_init(segment.module.projection_head, segment.module.projection_head_ema)
    else:
        segment.bank_init()
        segment.ema_init(segment.head, segment.head_ema)
        segment.ema_init(segment.projection_head, segment.projection_head_ema)

    ###################################################################################
    # First, run train_mediator.py
    path, is_exist = pickle_path_and_exist(args)

    # early save for time
    if is_exist:
        # load
        codebook = np.load(path)
        if args.distributed:
            segment.module.codebook.data = torch.from_numpy(codebook).cuda()
            segment.module.codebook.requires_grad = False
        else:
            segment.codebook.data = torch.from_numpy(codebook).cuda()
            segment.codebook.requires_grad = False

        # print successful loading modularity
        rprint(f'Modularity {path} loaded', rank)

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()

    else:
        rprint('Train Modularity-based Codebook First', rank)
        return
    ###################################################################################


    # optimizer
    optimizer = torch.optim.Adam(segment.parameters(), lr=1e-4 * ngpus_per_node)

    # tensorboard
    if (args.distributed == True) and (rank == 0):
        from datetime import datetime
        log_dir = os.path.join('logs',
                               datetime.today().strftime(" %m:%d_%H:%M")[2:],
                               args.dataset,
                               "_".join(
            [args.ckpt.split('/')[-1].split('.')[0],
             str(args.num_codebook),
             os.path.abspath(__file__).split('/')[-1]]))
        check_dir(log_dir)
    writer = SummaryWriter(log_dir=log_dir) if (rank == 0) and (args.distributed == True) else None

    # evaluation
    nice = NiceTool(args.n_classes)


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
            optimizer,
            writer, rank)


        test(
            epoch, # for decorator
            rank, # for decorator
            args,
            net.module if args.distributed else net,
            segment.module if args.distributed else segment,
            nice,
            test_loader,
            writer,
            rank)

        if (rank == 0):
            x = segment.module.state_dict() if args.distributed else segment.state_dict()
            baseline = args.ckpt.split('/')[-1].split('.')[0]

            # filepath hierarchy
            check_dir(f'CUSS/{args.dataset}/{baseline}/{args.num_codebook}')

            # save path
            y = f'CUSS/{args.dataset}/{baseline}/{args.num_codebook}/best_cnn.pth'
            torch.save(x, y)
            print(f'-----------------TEST Epoch {epoch}: BEST SAVING CHECKPOINT IN {y}-----------------')

        # Interrupt for sync GPU Process
        if args.distributed: dist.barrier()


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='cocostuff27', type=str)
    parser.add_argument('--ckpt', default='checkpoint/dino_vit_base_16.pth', type=str)
    parser.add_argument('--epoch', default=1, type=int)
    parser.add_argument('--distributed', default=True, type=str2bool)
    parser.add_argument('--load_Best', default=False, type=str2bool)
    parser.add_argument('--load_Fine', default=False, type=str2bool)
    parser.add_argument('--train_resolution', default=224, type=int)
    parser.add_argument('--test_resolution', default=320, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)

    # DDP
    parser.add_argument('--gpu', default='0,1,2,3', type=str)
    parser.add_argument('--port', default='12355', type=str)

    # parameter
    parser.add_argument('--grid', default='yes', type=str2bool)
    parser.add_argument('--num_codebook', default=2048, type=int)

    args = parser.parse_args()

    if 'dinov2' in args.ckpt: args.test_resolution=322

    if 'small' in args.ckpt:
        args.dim = 384
        args.reduced_dim=70
        args.projection_dim=2048

    elif 'base' in args.ckpt:
        args.dim = 768
        args.reduced_dim=70
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