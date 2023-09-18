import argparse
import os, sys
import time

import random
import json
import numpy as np

import torch
from torch.optim import lr_scheduler
import torch.optim
import torch.utils.data
from torch.nn import functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
from torch.utils.data.distributed import DistributedSampler
from sklearn import metrics
from torch.utils.tensorboard import SummaryWriter

import _init_paths
from cpg.bgm import BGM, BigJointDiscriminator
from lib.dataset.get_dataset import get_datasets, TransformUnlabeled_WS

from lib.dataset.handlers import  COCO2014_mask_handler, VOC2012_mask_handler, NUS_WIDE_mask_handler

from lib.models.MLDResnet import resnet50_ml_decoder
from lib.models.Resnet import create_model

from lib.utils.logger import setup_logger
from lib.utils.meter import AverageMeter, AverageMeterHMS, ProgressMeter
from lib.utils.helper import clean_state_dict, function_mAP, get_raw_dict, ModelEma, add_weight_decay, \
    get_learning_rate, get_port, str2bool, send_model_cuda
from lib.utils.losses import AsymmetricLoss

np.set_printoptions(suppress=True, precision=4)

MASK_HANDLER_DICT = {'voc': VOC2012_mask_handler, 'coco': COCO2014_mask_handler, 'nus': NUS_WIDE_mask_handler}

NUM_CLASS = {'voc': 20, 'coco': 80, 'nus': 81}

def parser_args():
    parser = argparse.ArgumentParser(description='Main')

    # data
    parser.add_argument('--dataset_name', default='voc', choices=['voc', 'coco', 'nus'],
                        help='dataset name')
    parser.add_argument('--dataset_dir',  default='../data', metavar='DIR',
                        help='dir of all datasets')
    parser.add_argument('--img_size', default=256, type=int,
                        help='size of input images')
    parser.add_argument('--output', default='./outputs', metavar='DIR', 
                        help='path to output folder')

    # training settings for warmup
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('--epochs', default=40, type=int, metavar='N',
                        help='number of total epochs to run')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--finetune', default=True, type=str2bool, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--warmup_batch_size', default=32, type=int,
                        help='batch size for warmup')
    parser.add_argument('--bs_ratio', default=4, type=int,
                        help='magnification of batchsize')
    parser.add_argument('--lr', '--learning_rate', default=1e-4, type=float, metavar='LR', 
                        help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight_decay', default=1e-4, type=float,metavar='W', 
                        help='weight decay (default: 1e-2)', dest='weight_decay')
    parser.add_argument('-p', '--print_freq', default=400, type=int, metavar='N', 
                        help='print frequency (default: 10)')
    parser.add_argument('--amp', action='store_true', default=True,
                        help='apply amp')
    parser.add_argument('--early_stop', action='store_true', default=False,
                        help='apply early stop')
    parser.add_argument('--save_PR', action='store_true', default=False,
                        help='on/off PR')
    parser.add_argument('--optim', default='adamw', type=str,
                        help='optimizer used')
    parser.add_argument('--warmup_epochs', default=10, type=int,
                        help='the number of epochs for warmup')
    parser.add_argument('--lb_ratio', default=0.05, type=float,
                        help='the ratio of lb:(lb+ub)')
    parser.add_argument('--loss_lb', default='asl', type=str, 
                        help='used_loss for lb')
    parser.add_argument('--loss_ub', default='bce', type=str,
                        help='used_loss for ub')
    parser.add_argument('--cutout', default=0.5, type=float, 
                        help='cutout factor')

    # Training settings
    parser.add_argument('--lr_g', type=float, default=1e-4)
    parser.add_argument('--lr_e', type=float, default=1e-4)
    parser.add_argument('--lr_d', type=float, default=1e-4)
    parser.add_argument('--lr_p', type=float, default=5e-5, help='lr of SCM prior network')
    parser.add_argument('--lr_a', type=float, default=1e-4, help='lr of adjacency matrix')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--d_steps_per_iter', type=int, default=1, help='how many D updates per iteration')
    parser.add_argument('--g_steps_per_iter', type=int, default=1, help='how many G updates per iteration')

    # Model settings
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--enc_coef', type=float, default=1, help='coefficient of the loss encoder')
    parser.add_argument('--sup_coef', type=float, default=1, help='coefficient of the supervised regularizer')
    parser.add_argument('--ub_coef', type=float, default=1, help='coefficient of the consistency of unlabled data')
    parser.add_argument('--sup_prop', type=float, default=1, help='proportion of supervised labels')
    parser.add_argument('--sup_type', type=str, default='ce', choices=['ce', 'l2'])
    parser.add_argument('--labels', type=str, default=None, help='name of the underlying structure')

    # Prior settings
    parser.add_argument('--prior', type=str, default='linscm', choices=['gaussian', 'uniform', 'linscm', 'nlrscm'],
                        help='latent prior p_z')

    # Encoder settings
    parser.add_argument('--enc_arch', type=str, default='resnet', choices=['resnet', 'resnet18', 'dcgan'],
                        help='encoder architecture')
    parser.add_argument('--enc_dist', type=str, default='gaussian', choices=['deterministic', 'gaussian', 'implicit'],
                        help='encoder distribution')
    parser.add_argument('--enc_fc_size', type=int, default=1024, help='number of nodes in fc layer of resnet')
    parser.add_argument('--enc_noise_dim', type=int, default=128)
    # Generator settings
    parser.add_argument('--dec_arch', type=str, default='sagan', choices=['sagan', 'dcgan'],
                        help='decoder architecture')
    parser.add_argument('--dec_dist', type=str, default='implicit', choices=['deterministic', 'gaussian', 'implicit'],
                        help='generator distribution')
    parser.add_argument('--g_conv_dim', type=int, default=32, help='base number of channels in encoder and generator')
    # Discriminator settings
    parser.add_argument('--dis_fc_size', type=int, default=512,
                        help='number of nodes in fc layer of joint discriminator')
    parser.add_argument('--d_conv_dim', type=int, default=32, help='base number of channels in discriminator')

    # # Output and save
    # parser.add_argument('--sample_every', type=int, default=1)
    # parser.add_argument('--sample_every_epoch', type=int, default=1)
    # parser.add_argument('--save_n_samples', type=int, default=64)
    # parser.add_argument('--save_n_recons', type=int, default=32)
    # parser.add_argument('--nrow', type=int, default=8)

    # model
    parser.add_argument('--ema_decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
    parser.add_argument('--resume', default=None, type=str,
                        help='path to latest checkpoint (default: none)')

    # multi-GPUs & Distributed Training
    parser.add_argument(
        "--world-size",
        default=1,
        type=int,
        help="number of nodes for distributed training",
    )
    parser.add_argument(
        "--rank", default=0, type=int, help="**node rank** for distributed training"
    )
    parser.add_argument(
        "-du",
        "--dist-url",
        default="tcp://127.0.0.1:11111",
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--seed", default=1, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--gpu", default=None, type=int, help="GPU id to use.")
    parser.add_argument(
        "--multiprocessing-distributed",
        type=str2bool,
        default=True,
        help="Use multi-processing distributed training to launch "
             "N processes per node, which has N GPUs. This is the "
             "fastest way to use PyTorch for either single node or "
             "multi node data parallel training",
    )


    args = parser.parse_args()

    if args.lb_ratio == 0.05:
        args.lb_bs = 2
        args.ub_bs = 6
    elif args.lb_ratio == 0.1:
        args.lb_bs = 2
        args.ub_bs = 6
    elif args.lb_ratio == 0.15:
        args.lb_bs = 2
        args.ub_bs = 6
    elif args.lb_ratio == 0.2:
        args.lb_bs = 2
        args.ub_bs = 6

    # get args net
    args.net = args.enc_arch
    # if args.enc_arch == 'resnet':
    #     args.net = 'resnet50'
    args.output = 'logs/' + args.net + '_outputs'
    args.resume = ('logs/%s_outputs/%s/%s/%s/warmup_%s_%s_%s/warmup_model.pth.tar'
                   %(args.net, args.dataset_name, args.img_size, args.lb_ratio, args.loss_lb, args.loss_ub, args.warmup_epochs))
    args.n_classes = NUM_CLASS[args.dataset_name]
    args.dataset_dir = os.path.join(args.dataset_dir, args.dataset_name) 
    
    args.output = os.path.join(args.output, args.dataset_name, '%s'%args.img_size,
                               '%s'%args.lb_ratio,  'CPG_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s'
                               %(args.loss_lb, args.loss_ub, args.warmup_epochs, args.lb_bs, args.epochs,
                                 args.lr_g, args.lr_e, args.lr_d, args.lr_p, args.lr_a, args.latent_dim,
                                 args.enc_coef, args.sup_coef, args.ub_coef))

    return args


def get_args():
    args = parser_args()
    return args


def main():
    args = get_args()

    # set distributed training
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)

    if args.gpu == "None":
        args.gpu = None

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):

    args.gpu = gpu
    # process.
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # save config and get logger
    os.makedirs(args.output, exist_ok=True)

    logger = setup_logger(output=args.output, distributed_rank=args.rank, color=False, name="XXX")
    logger.info("Command: " + ' '.join(sys.argv))

    path = os.path.join(args.output, "config.json")
    with open(path, 'w') as f:
        json.dump(get_raw_dict(args), f, indent=2)
    logger.info("Full config saved to {}".format(path))

    # build model
    logger.info(f"Use GPU: {args.gpu} for training")
    logger.info("Build Models")
    A = torch.ones((args.n_classes, args.n_classes))
    A = torch.triu(A, diagonal=1)
    model = BGM(args.latent_dim, args.g_conv_dim, args.img_size,
                args.enc_dist, args.enc_arch, args.enc_fc_size, args.enc_noise_dim, args.dec_dist,
                args.prior, args.n_classes, A)
    discriminator = BigJointDiscriminator(args.latent_dim, args.d_conv_dim, args.img_size,
                                          args.dis_fc_size)

    # build optimizer
    logger.info("Build Optimizer")
    if 'scm' in args.prior:
        enc_param = model.encoder.parameters()
        dec_param = list(model.decoder.parameters())
        prior_param = list(model.prior.parameters())
        A_optimizer = torch.optim.Adam(prior_param[0:1], lr=args.lr_a)
        prior_optimizer = torch.optim.Adam(prior_param[1:], lr=args.lr_p, betas=(args.beta1, args.beta2))
    else:
        enc_param = model.encoder.parameters()
        dec_param = model.decoder.parameters()
    encoder_optimizer = torch.optim.Adam(enc_param, lr=args.lr_e, betas=(args.beta1, args.beta2))
    decoder_optimizer = torch.optim.Adam(dec_param, lr=args.lr_g, betas=(args.beta1, args.beta2))
    D_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr_d, betas=(args.beta1, args.beta2))

    model = send_model_cuda(args, model)
    discriminator = send_model_cuda(args, discriminator, clip_batch=False)

    if args.resume:
        if os.path.exists(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(os.path.join(args.resume))

            args.start_epoch = args.warmup_epochs if args.finetune else checkpoint['epoch'] + 1
            if 'state_dict' in checkpoint and 'state_dict_ema' in checkpoint:
                state_dict = clean_state_dict(checkpoint['state_dict'])
                discriminator_state_dict = clean_state_dict(checkpoint['discriminator'])
            else:
                raise ValueError("No model or state_dict Found!!!")

            model.load_state_dict(state_dict, strict=False)
            discriminator.load_state_dict(discriminator_state_dict, strict=False)
            print(np.array(checkpoint['regular_mAP']))
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                .format(args.resume, checkpoint['epoch']))
            
            del checkpoint
            del state_dict
            # del state_dict_ema
            torch.cuda.empty_cache() 
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    ema_m = ModelEma(model, args.ema_decay)

    # Data loading code
    lb_train_dataset, ub_train_dataset, val_dataset = get_datasets(args)
    print("len(lb_train_dataset):", len(lb_train_dataset)) 
    print("len(ub_train_dataset):", len(ub_train_dataset))
    print("len(val_dataset):", len(val_dataset))

    lb_train_sampler = DistributedSampler(lb_train_dataset)
    lb_train_loader = torch.utils.data.DataLoader(
        lb_train_dataset, batch_size=args.lb_bs, num_workers=args.workers,
        drop_last=True, sampler=lb_train_sampler)

    ub_train_sampler = DistributedSampler(ub_train_dataset)
    ub_train_loader = torch.utils.data.DataLoader(
        ub_train_dataset, batch_size=args.ub_bs, num_workers=args.workers,
        drop_last=True, sampler=ub_train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=64, shuffle=False, num_workers=args.workers)

    # set scheduler for warmup
    args.steps_per_epoch = len(ub_train_loader)
    encoder_scheduler = lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=args.lr_e,
                                                steps_per_epoch=args.steps_per_epoch,
                                                epochs=args.epochs, pct_start=0.2)
    decoder_scheduler = lr_scheduler.OneCycleLR(decoder_optimizer, max_lr=args.lr_g,
                                                steps_per_epoch=args.steps_per_epoch,
                                                epochs=args.epochs, pct_start=0.2)
    D_scheduler = lr_scheduler.OneCycleLR(D_optimizer, max_lr=args.lr_d,
                                          steps_per_epoch=args.steps_per_epoch,
                                          epochs=args.epochs, pct_start=0.2)
    A_scheduler = lr_scheduler.OneCycleLR(A_optimizer, max_lr=args.lr_a,
                                          steps_per_epoch=args.steps_per_epoch,
                                          epochs=args.epochs, pct_start=0.2)
    prior_scheduler = lr_scheduler.OneCycleLR(prior_optimizer, max_lr=args.lr_p,
                                              steps_per_epoch=args.steps_per_epoch,
                                              epochs=args.epochs, pct_start=0.2)
    schedulers = {
        'encoder_scheduler': encoder_scheduler,
        'decoder_scheduler': decoder_scheduler,
        'D_scheduler': D_scheduler,
        'A_scheduler': A_scheduler,
        'prior_scheduler': prior_scheduler
    }

    epoch_time = AverageMeterHMS('TT')
    eta = AverageMeterHMS('ETA', val_only=True)
    mAPs = AverageMeter('mAP', ':5.5f', val_only=True)
    mAPs_ema = AverageMeter('mAP_ema', ':5.5f', val_only=True)
    progress = ProgressMeter(
        args.epochs,
        [eta, epoch_time, mAPs, mAPs_ema],
        prefix='=> Test Epoch: ')

    end = time.time()
    best_epoch = -1
    best_regular_mAP = 0
    best_regular_epoch = -1
    best_ema_mAP = 0
    regular_mAP_list = []
    ema_mAP_list = []
    best_mAP = 0


    # Used loss
    if args.loss_lb == 'bce':
        criterion_lb = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_lb == 'asl':
        criterion_lb = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')

    if args.loss_ub == 'bce':
        criterion_ub = torch.nn.BCEWithLogitsLoss(reduction='none')
    elif args.loss_ub == 'asl':
        criterion_ub = AsymmetricLoss(gamma_neg=4, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True, reduction='none')


    # tensorboard
    if not args.distributed or (args.distributed and args.rank == 0):
        summary_writer = SummaryWriter(log_dir=args.output)
    else:
        summary_writer = None

    torch.cuda.empty_cache()
    for epoch in range(args.start_epoch, args.epochs):
        lb_train_sampler.set_epoch(epoch)
        ub_train_sampler.set_epoch(epoch)
        torch.cuda.empty_cache()

        # train for one epoch
        if epoch < args.warmup_epochs:
            logger.info("Warmup epoch: {}".format(epoch))
            loss_disc_recoder, loss_encoder_recoder, loss_sup_recoder, loss_decoder_recoder \
                = warmup_train(lb_train_loader, ub_train_loader, model, discriminator, ema_m,
                             D_optimizer, encoder_optimizer, prior_optimizer, decoder_optimizer, A_optimizer,
                             schedulers, epoch, args, logger, criterion_lb, criterion_ub)
            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('loss_disc', loss_disc_recoder, epoch)
                summary_writer.add_scalar('loss_encoder', loss_encoder_recoder, epoch)
                summary_writer.add_scalar('loss_sup', loss_sup_recoder, epoch)
                summary_writer.add_scalar('loss_decoder', loss_decoder_recoder, epoch)
        else:
            if epoch == args.warmup_epochs:
                # del discriminator
                lb_train_loader = torch.utils.data.DataLoader(
                    lb_train_dataset, batch_size=args.lb_bs * args.bs_ratio, num_workers=args.workers,
                    drop_last=True, sampler=lb_train_sampler)

                ub_train_loader = torch.utils.data.DataLoader(
                    ub_train_dataset, batch_size=args.ub_bs * args.bs_ratio, num_workers=args.workers,
                    drop_last=True, sampler=ub_train_sampler)

                args.steps_per_epoch = len(ub_train_loader)
                encoder_scheduler = lr_scheduler.OneCycleLR(encoder_optimizer, max_lr=args.lr_e,
                                                            steps_per_epoch=args.steps_per_epoch,
                                                            epochs=args.epochs - args.warmup_epochs, pct_start=0.2)
                schedulers = {
                    'encoder_scheduler': encoder_scheduler,
                }

            logger.info("Semi training epoch: {}".format(epoch))
            loss_sup_recoder, loss_ub_recoder = semi_train(lb_train_loader, ub_train_loader, model, ema_m,
                                                         encoder_optimizer, schedulers, epoch, args, logger,
                                                         criterion_lb, criterion_ub)
            if summary_writer:
                # tensorboard logger
                summary_writer.add_scalar('loss_sup', loss_sup_recoder, epoch)
                summary_writer.add_scalar('loss_ub', loss_ub_recoder, epoch)



        # evaluate on validation set
        mAP = validate(val_loader, model, args, logger)
        mAP_ema = validate(val_loader, ema_m.module, args, logger)


        mAPs.update(mAP)
        mAPs_ema.update(mAP_ema)
        epoch_time.update(time.time() - end)
        end = time.time()
        eta.update(epoch_time.avg * (args.epochs - epoch - 1))

        regular_mAP_list.append(mAP)
        ema_mAP_list.append(mAP_ema)

        progress.display(epoch, logger)

        if summary_writer:
            # tensorboard logger
            summary_writer.add_scalar('val_mAP', mAP, epoch)
            summary_writer.add_scalar('val_mAP_ema', mAP_ema, epoch)


        # remember best (regular) mAP and corresponding epochs
        if mAP > best_regular_mAP:
            best_regular_mAP = max(best_regular_mAP, mAP)
            best_regular_epoch = epoch
        if mAP_ema > best_ema_mAP:
            best_ema_mAP = max(mAP_ema, best_ema_mAP)
            best_ema_epoch = epoch

        if mAP_ema > mAP:
            mAP = mAP_ema

        is_best = mAP > best_mAP
        if is_best and (not args.distributed or (args.distributed and args.rank == 0)):
            save_path = args.resume if epoch < args.warmup_epochs else os.path.join(args.output, 'best_model.pth.tar')
            dir_path = os.path.dirname(save_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            best_epoch = epoch
            best_mAP = mAP
            state_dict = model.state_dict()
            state_dict_ema = ema_m.module.state_dict()
            save_checkpoint({
                'epoch': epoch,
                'state_dict': state_dict,
                'state_dict_ema': state_dict_ema,
                'discriminator': discriminator.state_dict(),
                'regular_mAP': regular_mAP_list,
                'ema_mAP': ema_mAP_list,
                'best_regular_mAP': best_regular_mAP,
                'best_ema_mAP': best_ema_mAP,
                'encoder_optimizer': encoder_optimizer.state_dict(),
                'decoder_optimizer': decoder_optimizer.state_dict(),
                'D_optimizer': D_optimizer.state_dict(),
                'A_optimizer': A_optimizer.state_dict(),
                'prior_optimizer': prior_optimizer.state_dict(),
            }, is_best=True, filename=save_path)

        logger.info("{} | Set best mAP {} in ep {}".format(epoch, best_mAP, best_epoch))
        logger.info("   | best regular mAP {} in ep {}".format(best_regular_mAP, best_regular_epoch))

        # early stop

        if args.early_stop:
            if best_epoch >= 0 and epoch - max(best_epoch, best_regular_epoch) > 5:
                if len(ema_mAP_list) > 1 and ema_mAP_list[-1] < best_ema_mAP:
                    logger.info("epoch - best_epoch = {}, stop!".format(epoch - best_epoch))
                    break

    print("Best mAP:", best_mAP)

    if summary_writer:
        summary_writer.close()

    logger.info("Training is FINISHED".format(args.rank))


def set_optimizer(model, args):

    if args.optim == 'adam':
        parameters = add_weight_decay(model, args.weight_decay)
        optimizer = torch.optim.Adam(params=parameters, lr=args.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    elif args.optim == 'adamw':
        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad]},
        ]
        optimizer = getattr(torch.optim, 'AdamW')(
            param_dicts,
            args.lr,
            betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay
        )

    return optimizer

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    if is_best:
        torch.save(state, filename)


def warmup_train(lb_train_loader, ub_train_loader, model, discriminator, ema_m,
               D_optimizer, encoder_optimizer, prior_optimizer, decoder_optimizer, A_optimizer,
               schedulers, epoch, args, logger, criterion, criterion_ub):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    loss_disc_recoder = AverageMeter('L_disc', ':5.3f')
    loss_encoder_recoder = AverageMeter('L_encoder', ':5.3f')
    loss_decoder_recoder = AverageMeter('L_decoder', ':5.3f')
    loss_sup_recoder = AverageMeter('L_sup', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [lr, loss_disc_recoder, loss_encoder_recoder, loss_decoder_recoder, loss_sup_recoder, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # lr.update(get_learning_rate(optimizer))
    # logger.info("lr:{}".format(get_learning_rate(optimizer)))

    model.train()
    discriminator.train()
    lb_train_iter = iter(lb_train_loader)
    for i, ((inputs_w_ub, inputs_s_ub), _) in enumerate(ub_train_loader):

        # iteration for labeled training set
        try:
            (inputs_w_lb, _), labels_lb = next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (inputs_w_lb, _), labels_lb = next(lb_train_iter)

        num_lb = inputs_w_lb.shape[0]
        num_ub = inputs_w_ub.shape[0]
        batch_size = num_lb + num_ub
        # inputs_lb = torch.cat((inputs_w_lb, inputs_s_lb))
        # inputs_ub = torch.cat((inputs_w_ub, inputs_s_lb))
        inputs_w_lb = inputs_w_lb.cuda(args.gpu)
        inputs_w_ub = inputs_w_ub.cuda(args.gpu)
        inputs_s_ub = inputs_s_ub.cuda(args.gpu)
        inputs = torch.cat([inputs_w_lb, inputs_w_ub], dim=0).cuda(args.gpu)
        labels_lb = labels_lb.float().cuda(args.gpu)

        # ================== TRAIN DISCRIMINATOR ================== #
        for _ in range(args.d_steps_per_iter):
            discriminator.zero_grad()

            # Sample z from prior p_z
            if args.prior == 'uniform':
                z = torch.rand(inputs.size(0), args.latent_dim, device=inputs.device) * 2 - 1
            else:
                z = torch.randn(inputs.size(0), args.latent_dim, device=inputs.device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                # Get inferred latent z = E(x) and generated image x = G(z)
                if 'scm' in args.prior:
                    z_fake, x_fake, z, _ = model(inputs, z)
                else:
                    z_fake, x_fake, _ = model(inputs, z)

                # Compute D loss
                encoder_score = discriminator(inputs, z_fake.detach())
                decoder_score = discriminator(x_fake.detach(), z.detach())
                loss_d = F.softplus(decoder_score).mean() + F.softplus(-encoder_score).mean()

                del z_fake
                del x_fake

            scaler.scale(loss_d).backward()
            scaler.step(D_optimizer)
            scaler.update()

        # ================== TRAIN MODEL ================== #
        for _ in range(args.g_steps_per_iter):
            if args.prior == 'uniform':
                z = torch.rand(inputs.size(0), args.latent_dim, device=inputs.device) * 2 - 1
            else:
                z = torch.randn(inputs.size(0), args.latent_dim, device=inputs.device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                if 'scm' in args.prior:
                    z_fake, x_fake, z, z_fake_mean = model(inputs, z)
                else:
                    z_fake, x_fake, z_fake_mean = model(inputs, z)

                outputs_s_ub = model(inputs_s_ub, z)[-1][:, :args.n_classes]
                # ================== TRAIN ENCODER ================== #
                model.zero_grad()
                # WITH THE GENERATIVE LOSS
                encoder_score = discriminator(inputs, z_fake)
                loss_encoder = encoder_score.mean()

                # WITH THE SUPERVISED LOSS
                outputs = z_fake_mean[:num_lb, :args.n_classes]
                # sup_loss = criterion(outputs, labels_lb.repeat(2, 1)).mean()
                sup_loss = criterion(outputs, labels_lb).mean()
                con_loss = criterion_ub(outputs_s_ub, torch.sigmoid(z_fake_mean[num_lb:, :args.n_classes]).detach()).mean()
                loss_encoder = loss_encoder * args.enc_coef + sup_loss * args.sup_coef + con_loss * args.ub_coef

            scaler.scale(loss_encoder).backward()
            scaler.step(encoder_optimizer)
            # if 'scm' in args.prior:
            #     scaler.step(prior_optimizer)
            scaler.update()

            # ================== TRAIN GENERATOR ================== #
            if args.prior == 'uniform':
                z = torch.rand(inputs.size(0), args.latent_dim, device=inputs.device) * 2 - 1
            else:
                z = torch.randn(inputs.size(0), args.latent_dim, device=inputs.device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                if 'scm' in args.prior:
                    z_fake, x_fake, z, z_fake_mean = model(inputs, z)
                else:
                    z_fake, x_fake, z_fake_mean = model(inputs, z)

                model.zero_grad()

                decoder_score = discriminator(x_fake, z)
                # with scaling clipping for stabilization
                r_decoder = torch.exp(decoder_score.detach())
                s_decoder = r_decoder.clamp(0.5, 2)
                loss_decoder = -(s_decoder * decoder_score).mean()

            scaler.scale(loss_decoder).backward()
            scaler.step(decoder_optimizer)
            if 'scm' in args.prior:
                model.module.prior.set_zero_grad()
                scaler.step(A_optimizer)
                scaler.step(prior_optimizer)
            scaler.update()

        for scheduler in schedulers.values():
            scheduler.step()

        loss_disc_recoder.update(loss_d.item(), batch_size)
        loss_encoder_recoder.update(loss_encoder.item(), batch_size)
        loss_sup_recoder.update(sup_loss.item(), num_lb)
        loss_decoder_recoder.update(loss_decoder.item(), batch_size)
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)

    return loss_disc_recoder.avg, loss_encoder_recoder.avg, loss_sup_recoder.avg, loss_decoder_recoder.avg


def semi_train(lb_train_loader, ub_train_loader, model, ema_m, encoder_optimizer,
             schedulers, epoch, args, logger, criterion, criterion_ub):
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    loss_ub_recoder = AverageMeter('L_ub', ':5.3f')
    loss_sup_recoder = AverageMeter('L_super', ':5.3f')
    lr = AverageMeter('LR', ':.3e', val_only=True)
    mem = AverageMeter('Mem', ':.0f', val_only=True)
    progress = ProgressMeter(
        args.steps_per_epoch,
        [lr, loss_sup_recoder, loss_ub_recoder, mem],
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs))

    # lr.update(get_learning_rate(optimizer))
    # logger.info("lr:{}".format(get_learning_rate(optimizer)))

    model.train()
    for i, ((inputs_w_ub, inputs_s_ub), _) in enumerate(ub_train_loader):

        # iteration for labeled training set
        try:
            (inputs_w_lb, _), labels_lb = next(lb_train_iter)
        except:
            lb_train_iter = iter(lb_train_loader)
            (inputs_w_lb, _), labels_lb = next(lb_train_iter)

        num_lb = inputs_w_lb.shape[0]
        num_ub = inputs_w_ub.shape[0]
        batch_size = num_lb + num_ub
        inputs = torch.cat([inputs_w_lb, inputs_w_ub, inputs_s_ub], dim=0).cuda(args.gpu)
        labels_lb = labels_lb.float().cuda(args.gpu)

        with torch.cuda.amp.autocast(enabled=args.amp):
            z_infer = model(inputs)
            outputs = z_infer[:, :args.n_classes]
            loss_lb = criterion(outputs[:num_lb], labels_lb).mean()
            outputs_w_ub = outputs[num_lb:num_lb + num_ub]
            outputs_s_ub = outputs[num_lb + num_ub:]
            loss_ub = criterion_ub(outputs_s_ub, torch.sigmoid(outputs_w_ub).detach()).mean()
            loss = loss_lb + loss_ub * args.ub_coef

        model.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(encoder_optimizer)
        scaler.update()

        for scheduler in schedulers.values():
            scheduler.step()

        loss_sup_recoder.update(loss_lb.item(), batch_size)
        loss_ub_recoder.update(loss_ub.item(), batch_size)
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)
        ema_m.update(model)

        if i % args.print_freq == 0:
            progress.display(i, logger)
    return loss_sup_recoder.avg, loss_ub_recoder.avg

@torch.no_grad()
def validate(val_loader, model, args, logger):
    batch_time = AverageMeter('Time', ':5.3f')
    mem = AverageMeter('Mem', ':.0f', val_only=True)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, mem],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()
    outputs_sm_list = []
    targets_list = []
        
    end = time.time()
    for i, (inputs, targets) in enumerate(val_loader):
        inputs = inputs.cuda(args.gpu)
        targets = targets.cuda(args.gpu)

        # compute output
        with torch.cuda.amp.autocast(enabled=args.amp):
            z_infer = model(inputs)
            outputs = z_infer[:, :args.n_classes]
            outputs_sm = torch.sigmoid(outputs)
        
        # add list
        outputs_sm_list.append(outputs_sm.detach().cpu())
        targets_list.append(targets.detach().cpu())

        # record memory
        mem.update(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i, logger)

    labels = np.concatenate(targets_list)
    outputs = np.concatenate(outputs_sm_list)

    # calculate mAP
    mAP = function_mAP(labels, outputs)
    
    print("Calculating mAP:")  
    logger.info("  mAP: {}".format(mAP))

    return mAP


if __name__ == '__main__':
    main()