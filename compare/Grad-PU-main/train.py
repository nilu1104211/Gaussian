import os
import torch
import sys
sys.path.append(os.getcwd())
import time
from datetime import datetime
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dataset.dataset import PUDataset
from models.P2PNet import P2PNet
from args.pu1k_args import parse_pu1k_args
from args.pugan_args import parse_pugan_args
from args.utils import str2bool
from models.utils import *
import argparse


def batch_loss(args, model, points, targets, generator=None):
    points = rearrange(points, 'b n c -> b c n').contiguous().float().cuda()
    targets = rearrange(targets, 'b n c -> b c n').contiguous().float().cuda()
    interpolated = midpoint_interpolate(args, points)
    queries = get_query_points(interpolated, args, generator)
    return get_p2p_loss(args, model(interpolated, queries), queries, targets)


@torch.no_grad()
def validation_loss(args, model, loader):
    model.eval()
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    total = sum(len(points) * batch_loss(args, model, points, targets, generator).item()
                for points, targets, _ in loader)
    return total / len(loader.dataset)


def train(args):
    set_seed(args.seed)
    start = time.time()

    # load data
    train_dataset = PUDataset(args)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   shuffle=True,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.num_workers)
    best_only = getattr(args, 'best_only', False)
    val_h5 = getattr(args, 'val_h5_path', None)
    valid_loader = None
    if val_h5:
        valid_args = argparse.Namespace(**vars(args))
        valid_args.h5_file_path = val_h5
        valid_args.use_random_input = False
        valid_loader = torch.utils.data.DataLoader(PUDataset(valid_args, augment=False),
                                                   batch_size=args.batch_size)
    elif best_only:
        raise ValueError('--best_only needs --val_h5_path')

    # set up folders for checkpoints and logs
    str_time = datetime.now().isoformat()
    output_dir = os.path.join(args.out_path, str_time)
    ckpt_dir = os.path.join(output_dir, 'ckpt')
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
    log_dir = os.path.join(output_dir, 'log')
    writer = None if best_only else SummaryWriter(log_dir)
    logger = get_logger('train', None if best_only else log_dir)
    logger.info('Experiment ID: %s' % (str_time))

    # create model
    logger.info('========== Build Model ==========')
    model = P2PNet(args)
    model = model.cuda()
    # get the parameter size
    para_num = sum([p.numel() for p in model.parameters()])
    logger.info("=== The number of parameters in model: {:.4f} K === ".format(float(para_num / 1e3)))
    # log
    logger.info(args)
    logger.info(repr(model))
    # set model state
    model.train()

    # optimizer
    assert args.optim in ['adam', 'sgd']
    if args.optim == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    # lr scheduler
    scheduler_steplr = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.gamma)

    # train
    logger.info('========== Begin Training ==========')
    best_loss = float('inf')
    for epoch in range(args.epochs):
        model.train()
        logger.info('********* Epoch %d *********' % (epoch + 1))
        # epoch loss
        epoch_loss = 0.0

        for i, (input_pts, gt_pts, _) in enumerate(train_loader):
            loss = batch_loss(args, model, input_pts, gt_pts)
            epoch_loss += loss.item()

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # log
            if writer is not None:
                writer.add_scalar('train/loss', loss, i)
                writer.flush()
            if (i+1) % args.print_rate == 0:
                logger.info("epoch: %d/%d, iters: %d/%d, loss: %f" %
                      (epoch + 1, args.epochs, i + 1, len(train_loader), epoch_loss / (i+1)))

        # lr scheduler
        if not np.isfinite(epoch_loss):
            raise FloatingPointError(f'Nonfinite loss at epoch {epoch + 1}')
        scheduler_steplr.step()

        # log
        interval = time.time() - start
        logger.info("epoch: %d/%d, avg epoch loss: %f, time: %d mins %.1f secs" %
          (epoch + 1, args.epochs, epoch_loss / len(train_loader), interval / 60, interval % 60))

        if valid_loader is not None:
            value = validation_loss(args, model, valid_loader)
            if not np.isfinite(value):
                raise FloatingPointError(f'Nonfinite validation P2P loss at epoch {epoch + 1}')
            logger.info('Validation P2P loss: %.6f', value)
            if value < best_loss:
                best_loss = value
                best_path = os.path.join(ckpt_dir, 'ckpt-best.pth')
                torch.save(model.state_dict(), best_path + '.tmp')
                os.replace(best_path + '.tmp', best_path)
                logger.info('Best checkpoint updated at epoch %d: %s', epoch + 1, best_path)

        if not best_only and ((epoch + 1) % args.save_rate == 0 or epoch + 1 == args.epochs):
            model_name = 'ckpt-epoch-%d.pth' % (epoch+1)
            model_path = os.path.join(ckpt_dir, model_name)
            torch.save(model.state_dict(), model_path)

    if writer is not None:
        writer.close()


def parse_train_args():
    parser = argparse.ArgumentParser(description='Training Arguments')

    parser.add_argument('--dataset', default='pu1k', type=str, help='pu1k or pugan')
    parser.add_argument('--optim', default='adam', type=str, help='optimizer, adam or sgd')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--epochs', default=60, type=int, help='training epochs')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--print_rate', default=200, type=int, help='loss print frequency in each epoch')
    parser.add_argument('--save_rate', default=10, type=int, help='model save frequency')
    parser.add_argument('--out_path', default='./output', type=str, help='the checkpoint and log save path')

    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    train_args = parse_train_args()
    assert train_args.dataset in ['pu1k', 'pugan']

    if train_args.dataset == 'pu1k':
        model_args = parse_pu1k_args()
    else:
        model_args = parse_pugan_args()

    reset_model_args(train_args, model_args)

    train(model_args)
