import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import DeepLesionDataset as Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--data", default='/data/pacole2/DeepLesionPreprocessed/', help='Directory where data is stored')
parser.add_argument("--batch", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def main():
    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(root=opt.data, crop_size=(64, 64))
    # dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.outf)
    step = 0
    # noiseL_B=[0,55] # ingnored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_lr, img_hr = data
            img_lr, img_hr = img_lr.cuda(), img_hr.cuda()
            learned_noise = model(img_lr)
            loss = criterion(learned_noise, img_lr - img_hr) / (img_hr.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            learned_img = torch.clamp(img_lr - learned_noise, 0., 1.)
            psnr_train = batch_PSNR(learned_img, img_hr, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## the end of each epoch
        model.eval()
        # validate
        # psnr_val = 0
        # for k in range(len(dataset_val)):
        #     img_val = torch.unsqueeze(dataset_val[k], 0)
        #     noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
        #     imgn_val = img_val + noise
        #     img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
        #     out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
        #     psnr_val += batch_PSNR(out_val, img_val, 1.)
        # psnr_val /= len(dataset_val)
        # print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        # out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    main()
