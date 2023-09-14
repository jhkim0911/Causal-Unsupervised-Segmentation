import argparse

from tqdm import tqdm
from utils.utils import *
from modules.segment_module import transform, untransform
from loader.dataloader import dataloader
from torch.cuda.amp import autocast
from loader.netloader import network_loader, segment_mlp_loader, cluster_mlp_loader


def test(args, net, segment, cluster, nice, manytoone, test_loader, cmap):
    segment.eval()

    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    with Pool(40) as pool:
        for _, batch in prog_bar:
            # image and label and self supervised feature
            ind = batch["ind"].cuda()
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            with autocast():
                # intermediate feature
                feat = net(img)[:, 1:, :]
                feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
            seg_feat = transform(segment.head_ema(feat))
            seg_feat_flip = transform(segment.head_ema(feat_flip))
            seg_feat = untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

            # interp feat
            interp_seg_feat = F.interpolate(transform(seg_feat), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster preds
            cluster_preds = cluster.forward_centroid(untransform(interp_seg_feat), crf=True)

            # crf
            crf_preds = do_crf(pool, img, cluster_preds).argmax(1).cuda()

            # many to one matching
            manytoone.eval(crf_preds, label)
            mathcing_crf_preds = manytoone.do_hungarian(crf_preds)

            # nice evaluation
            _, desc_nice = nice.eval(mathcing_crf_preds, label)

            # save images
            save_all(args, ind, img, label, cluster_preds.argmax(dim=1), crf_preds, mathcing_crf_preds, cmap)

            # real-time print
            desc = f'{desc_nice}'
            prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()
    manytoone.reset()



def test_without_crf(args, net, segment, cluster, nice, manytoone, test_loader):
    segment.eval()

    total_acc = 0
    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    for idx, batch in prog_bar:
        # image and label and self supervised feature
        ind = batch["ind"].cuda()
        img = batch["img"].cuda()
        label = batch["label"].cuda()

        # intermediate feature
        with autocast():

            feat = net(img)[:, 1:, :]
            seg_feat_ema = segment.head_ema(feat)

            # linear probe loss
            linear_logits = segment.linear(seg_feat_ema)
            linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
            flat_label = label.reshape(-1)
            flat_label_mask = (flat_label >= 0) & (flat_label < args.n_classes)

            # interp feat
            interp_seg_feat = F.interpolate(transform(seg_feat_ema), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster
            cluster_preds = cluster.forward_centroid(untransform(interp_seg_feat), inference=True)

            # many to one matching
            manytoone.eval(cluster_preds, label)
            mathcing_cluster_preds = manytoone.do_hungarian(cluster_preds)

            # nice evaluation
            _, desc_nice = nice.eval(mathcing_cluster_preds, label)

        # linear probe acc check
        pred_label = linear_logits.argmax(dim=1)
        flat_pred_label = pred_label.reshape(-1)
        acc = (flat_pred_label[flat_label_mask] == flat_label[flat_label_mask]).sum() / flat_label[
            flat_label_mask].numel()
        total_acc += acc.item()

        # real-time print
        desc = f'[TEST] Acc (Linear): {100. * total_acc / (idx + 1):.1f}% | {desc_nice}'
        prog_bar.set_description(desc, refresh=True)

    # evaludation metric reset
    nice.reset()
    manytoone.reset()


def main(rank, args):

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # print argparse
    print_argparse(args, rank=0)

    # dataset loader
    _, test_loader, _ = dataloader(args, False)

    # network loader
    net = network_loader(args, rank)
    segment = segment_mlp_loader(args, rank)
    args.n_classes *= args.x_times 
    cluster = cluster_mlp_loader(args, rank)
    args.n_classes /= args.x_times
    
    # evaluation
    manytoone = ManyToOneMatching(args.n_classes, x_times=args.x_times)
    nice = NiceTool(args.n_classes)

    # color map
    cmap = create_cityscapes_colormap() if args.dataset == 'cityscapes' else create_pascal_label_colormap()

    # param size
    print(f'# of Parameters: {num_param(segment)/10**6:.2f}(M)') 

    # post-processing with crf and hungarian matching
    test_without_crf(
        args,
        net,
        segment,
        cluster,
        nice,
        manytoone,
        test_loader)

    # post-processing with crf and hungarian matching
    test(
        args,
        net,
        segment,
        cluster,
        nice,
        manytoone,
        test_loader,
        cmap)


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()
    
    # model parameter
    parser.add_argument('--NAME-TAG', default='CUSS-MLP', type=str)
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='cocostuff27', type=str)
    parser.add_argument('--port', default='12355', type=str)
    parser.add_argument('--ckpt', default='checkpoint/dino_vit_small_16.pth', type=str)
    parser.add_argument('--distributed', default=False, type=str2bool)
    parser.add_argument('--load_segment', default=True, type=str2bool)
    parser.add_argument('--load_cluster', default=True, type=str2bool)
    parser.add_argument('--train_resolution', default=320, type=int)
    parser.add_argument('--test_resolution', default=320, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)
    parser.add_argument('--gpu', default='4', type=str)
    parser.add_argument('--num_codebook', default=2048, type=int)

    # model parameter
    parser.add_argument('--reduced_dim', default=90, type=int)
    parser.add_argument('--projection_dim', default=2048, type=int)
    parser.add_argument('x_times', default=10, type=int)

    args = parser.parse_args()


    if 'dinov2' in args.ckpt:
        args.train_resolution=322
        args.test_resolution=322
    if 'small' in args.ckpt:
        args.dim=384
    elif 'base' in args.ckpt:
        args.dim=768
    args.num_queries=args.train_resolution**2 // int(args.ckpt.split('_')[-1].split('.')[0])**2
    

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    # first gpu index is activated once there are several gpu in args.gpu
    main(rank=gpu_list[0], args=args)
