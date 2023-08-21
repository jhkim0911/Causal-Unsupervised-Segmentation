import argparse

from tqdm import tqdm
from utils.utils import *
from loader.stego_dataloader import stego_dataloader
from torch.cuda.amp import autocast
from loader.netloader import network_loader, segment_cnn_loader


def test(args, net, segment, nice, test_loader, cmap):
    segment.eval()

    prog_bar = tqdm(enumerate(test_loader), total=len(test_loader), leave=True)
    with Pool(40) as pool:
        for idx, batch in prog_bar:
            # image and label and self supervised feature
            ind = batch["ind"].cuda()
            img = batch["img"].cuda()
            label = batch["label"].cuda()

            with autocast():
                # intermediate feature
                feat = net(img)[:, 1:, :]
                feat_flip = net(img.flip(dims=[3]))[:, 1:, :]
            seg_feat = segment.transform(segment.head_ema(feat))
            seg_feat_flip = segment.transform(segment.head_ema(feat_flip))
            seg_feat_ema = segment.untransform((seg_feat + seg_feat_flip.flip(dims=[3])) / 2)

            # interp feat
            interp_seg_feat = F.interpolate(segment.transform(seg_feat_ema), label.shape[-2:], mode='bilinear', align_corners=False)

            # cluster preds
            cluster_preds = segment.forward_centroid(segment.untransform(interp_seg_feat), inference=True)

            # crf
            onehot = F.one_hot(cluster_preds.to(torch.int64), args.n_classes).to(torch.float32)
            crf_preds = do_crf(pool, img, onehot.permute(0, 3, 1, 2)).argmax(1).cuda()

            # nice evaluation
            _, desc_nice = nice.eval(crf_preds, label)

            # hungarian
            hungarian_preds = nice.do_hungarian(crf_preds)

            # save images
            save_all(args, ind, img, label, cluster_preds, crf_preds, hungarian_preds, cmap, is_detr=False)

            # real-time print
            desc = f'{desc_nice}'
            prog_bar.set_description(desc, refresh=True)

def main(rank, args):

    # setting gpu id of this process
    torch.cuda.set_device(rank)

    # print argparse
    print_argparse(args, rank=0)

    # dataset loader
    _, test_loader, sampler = stego_dataloader(args, False)

    # network loader
    net = network_loader(args, rank)
    segment = segment_cnn_loader(args, rank)

    # evaluation
    nice = NiceTool(args.n_classes)

    # color map
    cmap = create_cityscapes_colormap() if args.dataset == 'cityscapes' else create_pascal_label_colormap()
    
    # param size
    print(f'# of Parameters: {segment.num_param/10**6:.2f}(M)') 


    # post-processing with crf and hungarian matching
    test(
        args,
        net.module if args.distributed else net,
        segment.module if args.distributed else segment,
        nice,
        test_loader,
        cmap)


if __name__ == "__main__":

    # fetch args
    parser = argparse.ArgumentParser()
    # model parameter
    parser.add_argument('--data_dir', default='/mnt/hard2/lbk-iccv/datasets', type=str)
    parser.add_argument('--dataset', default='cityscapes', type=str)
    parser.add_argument('--port', default='12355', type=str)
    parser.add_argument('--load_Best', default=False, type=str2bool)
    parser.add_argument('--load_Fine', default=True, type=str2bool)
    parser.add_argument('--ckpt', default='checkpoint/dino_vit_small_16.pth', type=str)
    parser.add_argument('--distributed', default=False, type=str2bool)
    parser.add_argument('--train_resolution', default=224, type=int)
    parser.add_argument('--test_resolution', default=320, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--num_workers', default=int(os.cpu_count() / 8), type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_codebook', default=2048, type=int)

    # model parameter
    parser.add_argument('--reduced_dim', default=70, type=int)
    parser.add_argument('--projection_dim', default=2048, type=int)

    args = parser.parse_args()

    if 'dinov2' in args.ckpt: args.test_resolution=322
    if 'small' in args.ckpt:
        args.dim = 384
    elif 'base' in args.ckpt:
        args.dim = 768

    # the number of gpus for multi-process
    gpu_list = list(map(int, args.gpu.split(',')))
    ngpus_per_node = len(gpu_list)

    # first gpu index is activated once there are several gpu in args.gpu
    main(rank=gpu_list[0], args=args)
