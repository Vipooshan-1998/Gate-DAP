from config.config import args
from net import GateDAP
import torch
from torch.utils.data import DataLoader
from data_load import LoadData
import os
import logging
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import time
from loss_function import Loss


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    # global loss
    model.train().to(device)
    start = time.time()
    for i, (input, target, _) in enumerate(tqdm(train_loader)):
        # print(time.time()-start)
        input = input.to(device)
        target = target.to(device)

        output = model(input)

        loss, kl_loss, nss_loss, cc_loss = criterion(target, output)

        # print(kl_loss(output, target))
        losses.update(loss.item(), target.size(0))
        # print(losses)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 10 == 0:
            print('Training Epoch:{:03d} Iter:{:03d} Loss:{:.6f} kl_loss:{:.6f} '
                  'nss_loss:{:.6f} cc_loss:{:.6f} in {:.3f}s'.\
                  format(epoch + 1, i + 1, losses.avg, kl_loss.item(),
                         nss_loss.item(), cc_loss.item(), time.time() - start))
        if (i + 1) % 1000 == 0:
            started = time.time()
            msg = 'Training Epoch:{:03d} Iter:{:03d} Loss:{:.6f} kl_loss:{:.6f} ' \
                  'nss_loss:{:.6f} cc_loss:{:.6f} in {:.3f}s'\
                .format(epoch + 1, i + 1, losses.avg, kl_loss.item(),
                       nss_loss.item(), cc_loss.item(), time.time() - start)
            logging.info(msg)
            started = time.time()
            print(msg)

    return losses.avg


def validate(valid_loader, model, criterion, epoch):
    losses = AverageMeter()
    model.eval().to(device)
    start = time.time()
    for i, (input, target, _) in enumerate(tqdm(valid_loader)):
        with torch.no_grad():
            input = input.to(device)
            target = target.to(device)
            # compute output
            output = model(input)
            loss, kl_loss, nss_loss, cc_loss = criterion(target, output)
            # measure accuracy and record loss
            losses.update(loss.item(), target.size(0))
            if (i + 1) % 10 == 0:
                print('Validating Epoch:{:03d} Iter:{:03d} Loss:{:.6f} kl_loss:{:.6f} '
                      'nss_loss:{:.6f} cc_loss:{:.6f} in {:.3f}s'.format(epoch + 1, i + 1, losses.avg, kl_loss.item(),
                             nss_loss.item(), cc_loss.item(), time.time() - start))
            if (i + 1) % 100 == 0:
                started = time.time()
                msg = 'Validating Epoch:{:03d} Iter:{:03d} Loss:{:.6f} kl_loss:{:.6f} ' \
                      'nss_loss:{:.6f} cc_loss:{:.6f} in {:.3f}s'.\
                    format(epoch + 1, i + 1, losses.avg, kl_loss.item(),
                           nss_loss.item(), cc_loss.item(), time.time() - start)
                logging.info(msg)
                started = time.time()
                print(msg)
    return losses.avg


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // (args.epochs // 1)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    name = 'attention'
    ckpts = 'ckpts/'  # save model
    if not os.path.exists(ckpts): os.makedirs(ckpts)
    log_file = os.path.join(ckpts + "/train_log_%s.txt" % (name,))
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', filename=log_file)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
    logging.getLogger('').addHandler(console)

    torch.cuda.manual_seed(2023)

    train_dataset = LoadData(model='train')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)
    val_dataset = LoadData(model='val')
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=True)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


    Model = GateDAP().to(device)
    checkpoint_model = torch.load(args.MAE_pretrained)
    # # checkpoint_model = checkpoint['model']
    state_dict = Model.state_dict()
    state_dict = {k: v for k, v in checkpoint_model.items() if k in state_dict and state_dict[k].size() == v.size()}
    Model.load_state_dict(state_dict, strict=False)
    best_loss = float('inf')
    # Model.load_state_dict(checkpoint_model['state_dict'])
    # args.start_epoch = checkpoint_model['epoch']
    # best_loss = checkpoint_model['valid_loss']
    # print(best_loss)

    optimizer = torch.optim.Adam(Model.parameters(), args.lr, weight_decay=args.weight_decay)
    # optimizer.load_state_dict(checkpoint_model['optim_dict'])
    # optimizer.load_state_dict(checkpoint_model['optim_dict'])
    criterion = Loss().to(device)
    logging.info('-------------- New training session, LR = %f ----------------' % (args.lr,))
    logging.info(
        '-- length of training images = %d--length of valid images = %d--'
        % (train_dataset.__len__(), val_dataset.__len__()))
    file_name = os.path.join(ckpts, 'model_best_%s.tar' % (name,))


    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train_loss = train(
            train_loader, Model, criterion, optimizer, epoch)

        valid_loss = validate(
            val_loader, Model, criterion, epoch)

        # remember best lost and save checkpoint
        best_loss = min(valid_loss, best_loss)
        file_name_last = os.path.join(ckpts, 'model_epoch_%d.tar' % (epoch + 1,))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': Model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'valid_loss': valid_loss,
        }, file_name_last)

        if valid_loss == best_loss:
            torch.save({
                'epoch': epoch + 1,
                'state_dict': Model.state_dict(),
                'optim_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'valid_loss': valid_loss,
            }, file_name)

        msg = 'Epoch: {:02d} Train loss {:.4f} | Valid loss {:.4f}'.format(
            epoch + 1, train_loss, valid_loss)
        logging.info(msg)
