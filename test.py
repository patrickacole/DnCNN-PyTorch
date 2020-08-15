import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--data", type=str, default='/data/pacole2/DeepLesionTestPreprocessed/', help='Directory where data is stored')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
opt = parser.parse_args()

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data
    print('Loading data ...\n')
    dataset_test = Dataset(root=opt.data, crop_size=None)
    loader_test = DataLoader(dataset=dataset_test, num_workers=4, batch_size=1, shuffle=False)
    # process data
    psnr_test = 0
    for i, (img_lr, img_hr) in enumerate(loader_test, 0):
        # image
        img_lr = img_lr.cuda()
        img_hr = img_hr.cuda()
        with torch.no_grad(): # this can save much memory
            learned_img = torch.clamp(img_lr - model(img_lr), 0., 1.)

        psnr = batch_PSNR(learned_img, img_hr, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))

        # TODO: write code to save images

    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
