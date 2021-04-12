import torch, copy
from torch import optim, nn
from torch.autograd import Variable

from .utils import models_copy, all_acc, myloss, fisher_information, set_bn_eval, print_model
from .standard import normal_train

def process(model, cut_idx, args, train_loader, test_loader, labels, result_file='./ewc.txt'):
    gpu = torch.device('cuda:0')
    new_model = model.cuda(gpu)
    models, cur_label = [copy.deepcopy(new_model) for _ in range(args.num_task)], []
    
    optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(args.num_task)]
    current_lr = args.lr / args.lr_adjust

    fishers = []
    for task in range(args.num_task):
        model, optimizer = models[task] if task == 0 else models_copy(models[task], models[task-1], cut_idx), optimizers[task] 
        # print_model(model)
        model.set_cut_idx(cut_idx)
        #model为tmp变量循环用，model：copy了上一次task中model的param（与cut_idx有关）
        current_lr *= args.lr_adjust
        print('Training Task {}... Labels: {}... Current LR: {}'.format(task, labels[task], current_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        cur_label = cur_label + labels[task] if args.class_incremental else labels[task]
        for epoch in range(args.epochs):
            if task == 0:
                loss = normal_train(model, cur_label, optimizer, train_loader[task], gpu)
            elif args.online:
                loss = ewc_train(model, cur_label, optimizer, train_loader[task], fishers[-1:], args.lam, gpu) #与normal_train相比多了ewcs[]与args.lam
            else:
                loss = ewc_train(model, cur_label, optimizer, train_loader[task], fishers, args.lam, gpu) #与online区别是ewcs[-1:]为ewcs只取最后一个元素构成的列表
            print('Epoch: {}\tLoss:{}'.format(epoch, loss))
            all_acc(task, models, test_loader, cut_idx, cur_label, args.class_incremental, labels, gpu, epoch, loss, result_file)
            if epoch % args.decay == 0 and epoch != 0:
                # current_lr *= 0.95
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
        fishers.append(fisher_information(model, cur_label, train_loader[task], cut_idx, gpu))


def ewc_train(model: nn.Module, labels: list, optimizer: torch.optim, 
                data_loader: torch.utils.data.DataLoader, fishers: list, lam: float, gpu: torch.device):
    model.train()
    # model.apply(set_bn_eval) #冻结BN及其统计数据
    epoch_loss = 0
    for data, target in data_loader:
        data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        new_target = target.clone()
        for idx, label in enumerate(labels):
            new_target[target==label] = idx
        loss = criterion(output[:, labels], new_target) 
        # loss = myloss(output, target, labels)
        for fisher in fishers:
            loss += (lam / 2) * fisher.penalty(model)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)