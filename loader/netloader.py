from utils.utils import *
from modules.segment import Segment
from modules.segment2 import Segment2
from torch.nn.parallel import DistributedDataParallel

def network_loader(args, rank=0):
    # load network
    net = load_model(args.ckpt, rank).cuda()
    if args.distributed:
        net = DistributedDataParallel(net, device_ids=[rank])
    freeze(net)
    return net

def segment_loader(args, rank=0):
    segment = Segment(args).cuda()

    if args.load_Best:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CUSS/{args.dataset}/{baseline}/{args.num_codebook}/best.pth'
        segment.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Best] {y} loaded', rank)
    elif args.load_Fine:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CUSS/{args.dataset}/{baseline}/{args.num_codebook}/finetune.pth'
        segment.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Fine] {y} loaded', rank)
    else:
        rprint('No Pretrained', rank)

    if args.distributed:
        segment = DistributedDataParallel(segment, device_ids=[rank])

    return segment

def segment2_loader(args, rank=0):
    segment = Segment2(args).cuda()

    if args.load_Best:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CUSS/{args.dataset}/{baseline}/{args.num_codebook}/best2.pth'
        segment.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Best] {y} loaded', rank)
    elif args.load_Fine:
        baseline = args.ckpt.split('/')[-1].split('.')[0]
        y = f'CUSS/{args.dataset}/{baseline}/{args.num_codebook}/finetune2.pth'
        segment.load_state_dict(torch.load(y, map_location=f'cuda:{rank}'), strict=False)
        rprint(f'[Fine] {y} loaded', rank)
    else:
        rprint('No Pretrained', rank)

    if args.distributed:
        segment = DistributedDataParallel(segment, device_ids=[rank])

    return segment


def load_model(ckpt, rank=0):
    # name and arch
    name = ckpt_to_name(ckpt)
    arch = ckpt_to_arch(ckpt)

    if name == "dino" or name == "mae":
        import models.dinomaevit as model
    elif name == "dinov2":
        import models.dinov2vit as model
    elif name == "mocov3":
        import models.mocov3vit as model
    else:
        raise ValueError

    net = getattr(model, arch)()
    checkpoint = torch.load(ckpt, map_location=torch.device(f'cuda:{rank}'))
    if name == "mae":
        msg = net.load_state_dict(checkpoint["model"], strict=False)
    elif name == "dino":
        msg = net.load_state_dict(checkpoint, strict=False)
    elif name == "dinov2":
        msg = net.load_state_dict(checkpoint, strict=False)
    elif name == "mocov3":
        def checkpoint_module(checkpoint, net):
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            msg = net.load_state_dict(new_state_dict, strict=False)
            return msg

        msg = checkpoint_module(checkpoint['state_dict'], net)

    # check incompatible layer or variables
    rprint(msg, rank)

    return net
