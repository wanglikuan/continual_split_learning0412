import torch, copy
from torch import optim, nn
from torch.autograd import Variable

from .utils import models_copy, all_acc, myloss, fisher_information
from .standard import normal_train

def process(model, cut_idx, args, train_loader, test_loader, labels, result_file='./our_process.txt'):
    gpu = torch.device('cuda:0')
    new_model = model.cuda(gpu)
    models, cur_label = [copy.deepcopy(new_model) for _ in range(args.num_task)], []
    
    optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(args.num_task)]
    current_lr = args.lr / args.lr_adjust

    fishers, if_freeze, loss = [], 0, 10

    for task in range(args.num_task):
        model, optimizer = models[task] if task == 0 else models_copy(models[task], models[task-1], cut_idx), optimizers[task] #model为tmp变量循环用，model：copy了上一次task中model的param（与cut_idx有关）
        current_lr *= args.lr_adjust
        print('Training Task {}... Labels: {}... Current LR: {}'.format(task, labels[task], current_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        cur_label = cur_label + labels[task] if args.class_incremental else labels[task]
        for epoch in range(args.epochs):
            if task == 0:
                loss = normal_train(model, cur_label, optimizer, train_loader[task], gpu)
            else:
                if args.online:
                    # loss = our_train(model, cur_label, optimizer, train_loader[task], ewcs[-1:], args.lam, gpu, cut_idx, args.threshold) 
                    # #与normal_train相比多了ewcs[]与args.lam
                    loss = our_train(model, cur_label, optimizer, train_loader[task], fishers[-1:], args.lam, gpu, cut_idx, if_freeze) 
                    #与normal_train相比多了ewcs[]与args.lam
                else:
                    # loss = our_train(model, cur_label, optimizer, train_loader[task], ewcs, args.lam, gpu, cut_idx, args.threshold) 
                    # #与online区别是ewcs[-1:]为ewcs只取最后一个元素构成的列表
                    loss = our_train(model, cur_label, optimizer, train_loader[task], fishers, args.lam, gpu, cut_idx, if_freeze) 
                    #与online区别是ewcs[-1:]为ewcs只取最后一个元素构成的列表
                
                #判断loss，loss若小于阈值，令变量if_freeze=1,传入下次our_train
                #our_train相比于ewc_train多两个参数：if_freeze和cut_idx
                if_freeze = 1 if loss < args.threshold else 0

            print('Epoch: {}\tLoss:{}'.format(epoch, loss))
            # print('Iteration: {}\tLoss:{}\tif freeze:{}'.format(iteration, loss, if_freeze))
            all_acc(task, models, test_loader, cut_idx, cur_label, args.class_incremental, labels, gpu, epoch, loss, result_file)
            if epoch % args.decay == 0 and epoch != 0:
                # current_lr *= 0.95
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.95
        fishers.append(fisher_information(model, cur_label, train_loader[task], cut_idx, gpu))

# def our_train(model: nn.Module, labels: list, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, fishers: list, lam: float, gpu: torch.device, cut_idx, threshold):
#     model.train()
#     # model.apply(set_bn_eval) #冻结BN及其统计数据
#     epoch_loss = 0
#     for data, target in data_loader:
#         data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
#         optimizer.zero_grad()
#         output = model(data)

#         criterion = nn.CrossEntropyLoss()
#         new_target = target.clone()
#         for idx, label in enumerate(labels):
#             new_target[target==label] = idx
#         loss = criterion(output[:, labels], new_target) 
#         # loss = myloss(output, target, labels)
#         for fisher in fishers:
#             loss += (lam / 2) * fisher.penalty(model)
#         epoch_loss += loss.item()

#         loss.backward()

#         if server_update:
#             optimizer.step()
#         else:
#             for group in optimizer.param_groups:
#                 for idx, p in enumerate(group['params']):
#                     if (idx < cut_idx) and (p.grad is not None):
#                         d_p = p.grad.data
#                         p.data.add_(-group['lr'], d_p)
#     return epoch_loss / len(data_loader)

# Implemented by Xiaosong Ma 
def our_train(model: nn.Module, labels: list, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, fishers: list, lam: float, gpu: torch.device, cut_idx, if_freeze):

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

        if if_freeze == 1 :
            for group in optimizer.param_groups:
                for idx, p in enumerate(group['params']):
                    if idx >= cut_idx or p.grad is None: #冻结server，即跳过cut_idx ~ end
                        continue                    
                    d_p = p.grad
                    p.data = p.data - d_p*group['lr']
        else:
            optimizer.step()

    return epoch_loss / len(data_loader)