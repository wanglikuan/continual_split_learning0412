import torch, copy
from torch import optim, nn
from torch.autograd import Variable

from .utils import test_model

def process(model, cut_idx, args, train_loader, test_loader, labels, result_file='./multitask.txt'):
    gpu = torch.device('cuda:0')
    new_model = model.cuda(gpu)
    # print_model(new_model)
    models = [copy.deepcopy(new_model) for _ in range(args.num_task)]

    optimizers = [optim.SGD(params=models[idx].parameters(), lr=args.lr) for idx in range(args.num_task)]
    current_lr = args.lr / args.lr_adjust

    for epoch in range(args.epochs):
        loss = 0.0
        for task in range(args.num_task):
            model, optimizer = models[task], optimizers[task]
            loss += (train(model, optimizer, train_loader[task], gpu) / args.num_task)
        print('Epoch: {}\tLoss: {}'.format(epoch, loss))
        temp_param = [torch.zeros_like(param.data) for param in models[0].parameters()]
        for task in range(args.num_task):
            for idx, param in enumerate(models[task].parameters()):
                if idx >= cut_idx:
                    temp_param[idx] += param.data/args.num_task
        
        for task in range(args.num_task):
            for idx, param in enumerate(models[task].parameters()):
                if idx >= cut_idx:
                    param.data = temp_param[idx]
        
        for task in range(args.num_task):
            acc = test_model(models[task], None, test_loader[task], gpu)
            print("Task: {}\t\tAccuracy: {}".format(task, acc))
            with open(result_file, 'a') as f:
                f.write('{}\t{}\t{}\t{}\n'.format(epoch, loss, task, acc))

def train(model:nn.Module, optimizer: torch.optim, data_loader: torch.utils.data.DataLoader, gpu: torch.device):
    model.train()
    epoch_loss = 0
    for data, target in data_loader:
        data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
        optimizer.zero_grad()
        output = model(data)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    return epoch_loss / len(data_loader)