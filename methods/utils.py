import copy, torch, math
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

def models_copy(target_model, src_model, cut_idx=0):
    temp_dict = src_model.state_dict()
    target_dict = target_model.state_dict()
    cut_label, flag = (list(dict(target_model.named_parameters()).keys()))[cut_idx], False
    for key in target_dict.keys():
        flag = max(flag, (key == cut_label))
        # print(key)
        if 'bn' in key:
            continue
        target_dict[key] = (torch.zeros_like(target_dict[key]) + temp_dict[key]) if flag else target_dict[key]
    target_model.load_state_dict(target_dict)
    return target_model

def test_model(model, labels, test_data, gpu):
    correct, total = 0, 0
    model.eval()
    with torch.no_grad():
        for data, target in test_data:
            data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
            output = model(data)
            if labels is not None:
                offset = torch.zeros_like(output) - math.inf
                offset[:, labels] = 0.0
                output = output + offset
            # get the index of the max log-probability
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc = correct / total
    return acc

def all_acc(task, models, test_loader, cut_idx, cur_label, class_incremental, labels, gpu, iteration, loss, result_file):
    for sub_task in range(task + 1):
        temp_model = copy.deepcopy(models[sub_task])
        temp_model = models_copy(temp_model, models[task], cut_idx)
        if iteration == 1:
            print("Task {}'s model".format(sub_task))
            print_model(temp_model)
        for i in range(task + 1):
            test_label = cur_label if class_incremental else labels[i]
            acc = test_model(temp_model, test_label, test_loader[i], gpu)
            print('Device Task: {}\tTest Task: {}\tAccuracy: {}'.format(sub_task, i, acc))
            with open(result_file, 'a') as f:
                # Current server parameter (task) + device parameter (task) --> training a given task
                f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(task, iteration, loss, sub_task, i, acc))

def myloss(output, target, labels, weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean'):
    def customized_log_softmax(output, dim, labels):
        if len(labels) == output.size(1):
            return output.log_softmax(1)
        temp = output.softmax(dim)
        for result in temp:
            a = 0.0
            for idx in labels:
                a += result[idx]
            result = result / a
        return torch.log(temp)
    return F.nll_loss(customized_log_softmax(output, 1, labels), target, weight, None, ignore_index, None, reduction)

def set_bn_eval(m):
    # print(m.__dict__)
    classname = m.__class__.__name__
    if classname.find('BatchNorm2d') != -1:
        m.eval()

def print_model(model:nn.Module):
    # for idx, (name, param) in enumerate(model.named_parameters()):
    #     print(idx, name)
    # print("===================")
    # for idx, (key, value) in enumerate(model.state_dict().items()):
    #     print(idx, key)
    print(model.state_dict())

class fisher_information(object):
    # Here we will store the optimal solution for a given task. 
    def __init__(self, model: nn.Module, labels: list, dataset: torch.utils.data.DataLoader, cut_idx: int, gpu: torch.device):
        self.cut_idx = cut_idx
        self.fisher = self.cal_fisher(model, labels, dataset, gpu)
        self.optpar = [param.data.clone() for param in model.parameters()]

    def cal_fisher(self, model, labels, dataset, gpu):
        model.train()
        losses, criterion = [], nn.CrossEntropyLoss()
        gradient = [torch.zeros_like(param) for param in model.parameters()]

        for data, target in dataset:
            model.zero_grad()
            data, target = Variable(data).cuda(gpu), Variable(target).cuda(gpu)
            new_target = target.clone()
            for idx, label in enumerate(labels):
                new_target[target==label] = idx
            loss = criterion(model(data)[:, labels], new_target)
            # losses.append(loss)
            loss.backward()
            for idx, param in enumerate(model.parameters()):
                gradient[idx] += (param.grad.data.clone() / len(dataset))
        
        # loss = torch.mean(losses)
        # loss.backward()

        return [g.pow(2) for g in gradient]

    def penalty(self, model: nn.Module):
        loss = 0
        for idx, param in enumerate(model.parameters()):
            if idx < self.cut_idx:
                continue
            _loss = self.fisher[idx] * (param - self.optpar[idx]).pow(2)
            loss += _loss.sum()
        # print('penalty:', loss)
        return loss