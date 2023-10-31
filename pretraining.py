import argparse
import os
import os.path as osp
import shutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model.models.classifier import Classifier
from model.dataloader.samplers import CategoriesSampler
from model.utils import pprint, set_gpu, ensure_path, Averager, Timer, count_acc, euclidean_metric
from tqdm import tqdm

#DONE # TODO - Compare implementation of the base models.
#DONE # TODO - Need to use AvgPool2d(5) for conv and AdaptiveAvgPool2d for ResNet
#DONT # TODO - Make sure dataset has same data augmentation
# TODO - Using MiniImageNet from Optimization as a Model for FSL.
# TODO - TieredImageNet from https://drive.google.com/file/d/1nVGCTd9ttULRXFezh4xILQ9lUkg0WZCG/view
# TODO - To train a model with WRN, a trick is to set fix_BN to True (set BN to eval mode) during the meta-learning stage,
#  which improves a lot. For ProtoNet, the temperature is important, which should be large, e.g., 64.
# TODO - Temperature on the encoder when used in Meta-Learning.


# TODO - MiniImageNet
# python pretrain.py --lr 0.001 --model conv --schedule 170 300 400 500 --gamma 0.1 --batch_size 128 --max_epoch 800
# python pretrain.py --lr 0.1 --model ResNet --schedule 350, 400, 440, 460, 480 --gamma 0.1 --batch_size 128 --max_epoch 500

# TODO - TieredImageNet
# python pretrain.py --lr 0.1 --batch_size 128 --max_epoch 600 --backbone_class Res12 --schedule 400 500 550 580 --ngpu 1 --gamma 0.1 --dataset TieredImagenet --query 10

# TODO - CUB Dataset
# python pretrain.py --lr 0.001 --batch_size 256 --max_epoch 500 --backbone_class ConvNet --schedule 300 350 400 450 --gamma 0.1 --dataset CUB --query 15
# python pretrain.py --lr 0.1 --batch_size 256 --max_epoch 600 --backbone_class ResNet --schedule 400, 500, 550, 580 --gamma 0.1 --dataset CUB --query 15


# DONE # TODO - 200 batches per epoch.

# pre-train model, compute validation acc after 500 epoches
if __name__ == '__main__':

    # Parsing in the pretraining arguments for the script.
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--dataset', type=str, default='MiniImageNet', choices=['MiniImageNet', 'TieredImagenet', 'CUB'])    
    parser.add_argument('--backbone_class', type=str, default='Res12', choices=['ConvNet', 'Res12'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[75, 150, 300], help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--query', type=int, default=15)    
    parser.add_argument('--resume', type=bool, default=False)
    args = parser.parse_args()
    args.orig_imsize = -1
    pprint(vars(args))
    
    save_path1 = '-'.join([args.dataset, args.backbone_class, 'Pre'])
    save_path2 = '_'.join([str(args.lr), str(args.gamma), str(args.schedule)])
    args.save_path = osp.join(save_path1, save_path2)
    if not osp.exists(save_path1):
        os.mkdir(save_path1)
    ensure_path(args.save_path)

    # Importing the desired dataset.
    if args.dataset == 'MiniImageNet':
        # Handle MiniImageNet
        from model.dataloader.mini_imagenet import MiniImageNet as Dataset
    elif args.dataset == 'CUB':
        from model.dataloader.cub import CUB as Dataset
    elif args.dataset == 'TieredImagenet':
        from model.dataloader.tiered_imagenet import tieredImageNet as Dataset    
    else:
        raise ValueError('Non-supported Dataset.')

    # Loading the desired dataset and samplers.
    trainset = Dataset('train', args, augment=True)
    train_loader = DataLoader(dataset=trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    args.num_class = trainset.num_class
    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, 200, valset.num_class, 1 + args.query) # test on 16-way 1-shot
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    args.way = valset.num_class
    args.shot = 1
    
    # Constructing the base model
    model = Classifier(args)

    # Constructing the optimizer, Adam for Conv and SGD for ResNet
    if 'Conv' in  args.backbone_class:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    elif 'Res' in args.backbone_class:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    else:
        raise ValueError('No Such Encoder') 

    # Creating the loss function (Cross Entropy Loss).   
    criterion = torch.nn.CrossEntropyLoss()
    
    # Sending the model to the correct device.
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        if args.ngpu  > 1:
            model.encoder = torch.nn.DataParallel(model.encoder, device_ids=list(range(args.ngpu)))
        
        model = model.cuda()
        criterion = criterion.cuda()
    
    # Function for saving the models state dictionary.
    def save_model(name):
        torch.save(dict(params=model.state_dict()), osp.join(args.save_path, name + '.pth'))
    
    # Function for saving model and state checkpoints.
    def save_checkpoint(is_best, filename='checkpoint.pth.tar'):
        state = {'epoch': epoch + 1,
                 'args': args,
                 'state_dict': model.state_dict(),
                 'optimizer' : optimizer.state_dict(),
                 'global_count': global_count}
        
        torch.save(state, osp.join(args.save_path, filename))
        if is_best:
            shutil.copyfile(osp.join(args.save_path, filename), osp.join(args.save_path, 'model_best.pth.tar'))
    
    # If we are loading in a partially trained model.
    if args.resume == True:
        # load checkpoint
        state = torch.load(osp.join(args.save_path, 'model_best.pth.tar'))
        init_epoch = state['epoch']
        resumed_state = state['state_dict']
        # resumed_state = {'module.'+k:v for k,v in resumed_state.items()}
        model.load_state_dict(resumed_state)
        optimizer.load_state_dict(state['optimizer'])
        initial_lr = optimizer.param_groups[0]['lr']
        global_count = state['global_count']
    
    # If we are starting a fresh training session.
    else:
        init_epoch = 1
        initial_lr = args.lr
        global_count = 0

    # Tracking training inforamtion.

    for epoch in range(init_epoch, args.max_epoch + 1):

        # Manually performing learning rate schedule updates.
        if epoch in args.schedule:
            initial_lr *= args.gamma
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr
        
        model.train()

        for i, batch in enumerate(train_loader, 1):
            global_count = global_count + 1

            # Sending the batch of data to the correct device.
            if torch.cuda.is_available():
                data, label = [_.cuda() for _ in batch]
                label = label.type(torch.cuda.LongTensor)
            else:
                data, label = batch
                label = label.type(torch.LongTensor)

            # Computing the forward pass and computing its loss.
            logits = model(data)
            loss = criterion(logits, label)

            # Monitoring the accuracy.
            acc = count_acc(logits, label)

            # Typical training stuff in PyTorch.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # do not do validation in first 500 epoches
        if epoch > 100 or (epoch-1) % 5 == 0:
            model.eval()

            # test performance with Few-Shot
            label = torch.arange(valset.num_class).repeat(args.query)

            # Sending to the correct device.
            if torch.cuda.is_available():
                label = label.type(torch.cuda.LongTensor)
            else:
                label = label.type(torch.LongTensor)   

            with torch.no_grad():
                for i, batch in tqdm(enumerate(val_loader, 1)):

                    # Sending the data to the correct device.
                    if torch.cuda.is_available():
                        data, _ = [_.cuda() for _ in batch]
                    else:
                        data, _ = batch

                    data_shot, data_query = data[:valset.num_class], data[valset.num_class:] # 16-way test
                    
                    logits_dist, logits_sim = model.forward_proto(data_shot, data_query, valset.num_class)

                    loss_dist = F.cross_entropy(logits_dist, label)
                    acc_dist = count_acc(logits_dist, label)
                    loss_sim = F.cross_entropy(logits_sim, label)
                    acc_sim = count_acc(logits_sim, label)               

            # If it was the best seen yet then save the model.
            if va_dist > trlog['max_acc_dist']:
                save_model('max_acc_dist')
                save_checkpoint(True)
                
            if va_sim > trlog['max_acc_sim']:
                save_model('max_acc_sim')
                save_checkpoint(True)            
                    
            save_model('epoch-last')
