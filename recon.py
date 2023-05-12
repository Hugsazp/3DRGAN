#!/usr/bin/env python

import argparse
import sys
import os

import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from model import Generator, Discriminator, FeatureExtractor

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='folder', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='./test/800', help='path to dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
# parser.add_argument('--high_resolution',type = int,default = 40,help = 'the high resolution image size')
parser.add_argument('--scale_factor',type = int,default = 4,help = 'low to high resolution scaling factor')
parser.add_argument('--imageSize', type=int, default=800, help='the low resolution image size')
parser.add_argument('--cuda', action='store_true',default=True, help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--generatorWeights', type=str, default='checkpoints/generator_final3.pth', help="path to generator weights (to continue training)")
parser.add_argument('--discriminatorWeights', type=str, default='checkpoints/discriminator_final3.pth', help="path to discriminator weights (to continue training)")

opt = parser.parse_args()
print(opt)

try:
    os.makedirs('outcome/800/high_res_fake')
    os.makedirs('outcome/800/high_res_real')
    os.makedirs('outcome/800/low_res')
except OSError:
    pass


if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.RandomCrop(opt.imageSize),
                                transforms.ToTensor()])

#normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            #transforms.Normalize(mean = [0.485, 0.456, 0.406],
                            #                   std = [0.229, 0.224, 0.225])
                            ])

# Equivalent to un-normalizing ImageNet (for correct visualization)
#unnormalize = transforms.Normalize(mean = [-2.118, -2.036, -1.804], std = [4.367, 4.464, 4.444])

if opt.dataset == 'folder':
    # folder dataset
    dataset = datasets.ImageFolder(root=opt.dataroot, transform=transform)
elif opt.dataset == 'cifar10':
    dataset = datasets.CIFAR10(root=opt.dataroot, download=True, train=False, transform=transform)
elif opt.dataset == 'cifar100':
    dataset = datasets.CIFAR100(root=opt.dataroot, download=True, train=False, transform=transform)
assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))
generator = Generator(16,opt.imageSize,opt.scale_factor)
if opt.generatorWeights != '':
    generator.load_state_dict(torch.load(opt.generatorWeights),strict=False)
print ("load generator success!")

discriminator = Discriminator()
if opt.discriminatorWeights != '':
    discriminator.load_state_dict(torch.load(opt.discriminatorWeights))
print ("load discriminator success!")

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print ("load feature success!")

content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

# if gpu is to be used
if opt.cuda:
    generator.cuda()
    discriminator.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()

noise1 = torch.randn(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise2= torch.normal(mean = 0.5,std = 0.25, size=(opt.batchSize, 3 ,opt.imageSize, opt.imageSize) )

noise2[noise2>0.8]=1; noise2[noise2<=0.8]= 0# 阈值0.9，二值化
low_res = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

print ('Test started...')
mean_generator_content_loss = 0.0
mean_generator_adversarial_loss = 0.0
mean_generator_total_loss = 0.0
mean_discriminator_loss = 0.0

# Set evaluation mode (not training)
generator.eval()
discriminator.eval()

x = torch.zeros(3, 80, 80, 1).cuda()
y = torch.zeros(3, 80, 80, 1).cuda()
z = torch.zeros(3, 40, 40, 1).cuda()

for i, data in enumerate(dataloader):
    # Generate data
    high_res_real, _ = data

    # Downsample images to low resolution
    for j in range(opt.batchSize):
        low_res[j] = scale(high_res_real[j])
        #high_res_real[j] = normalize(high_res_real[j])
    # Generate real and fake inputs
    if opt.cuda:
        high_res_real = Variable(high_res_real.cuda())
        high_res_fake = generator(Variable(low_res+noise2).cuda())
    else:
        high_res_real = Variable(high_res_real)
        high_res_fake = generator(Variable(low_res))
    
    ######### Test discriminator #########

    discriminator_loss = adversarial_criterion(discriminator(high_res_real), target_real) + \
                            adversarial_criterion(discriminator(Variable(high_res_fake.data)), target_fake)
    mean_discriminator_loss += discriminator_loss.item()

    ######### Test generator #########

    real_features = Variable(feature_extractor(high_res_real).data)
    fake_features = feature_extractor(high_res_fake)

    generator_content_loss = content_criterion(high_res_fake, high_res_real) + 0.006*content_criterion(fake_features, real_features)
    mean_generator_content_loss += generator_content_loss.item()
    generator_adversarial_loss = adversarial_criterion(discriminator(high_res_fake), target_real)
    mean_generator_adversarial_loss += generator_adversarial_loss.item()

    generator_total_loss = generator_content_loss + 1e-3*generator_adversarial_loss
    mean_generator_total_loss += generator_total_loss.item()

    ######### Status and display #########
    sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f' % (i, len(dataloader),
    discriminator_loss.item(), generator_content_loss.item(), generator_adversarial_loss.item(), generator_total_loss.item()))

    for j in range(opt.batchSize):
        save_image(high_res_fake.data[j], 'outcome/800/high_res_fake/' + str(i*opt.batchSize + j).zfill(4) + '.png')
        save_image(high_res_real.data[j], 'outcome/800/high_res_real/' + str(i*opt.batchSize + j).zfill(4) + '.png')
        save_image(low_res[j]+noise2[j], 'outcome/800/low_res/' + str(i*opt.batchSize + j).zfill(4) + '.png')

sys.stdout.write('\r[%d/%d] Discriminator_Loss: %.4f Generator_Loss (Content/Advers/Total): %.4f/%.4f/%.4f\n' % (i, len(dataloader),
mean_discriminator_loss/len(dataloader), mean_generator_content_loss/len(dataloader), 
mean_generator_adversarial_loss/len(dataloader), mean_generator_total_loss/len(dataloader)))


