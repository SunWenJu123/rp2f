import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import relu, avg_pool2d
from typing import List, Tuple, Optional
from torch.nn.modules.utils import _pair
from torch.nn import Parameter, init
from torch import Tensor
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR

from backbone.MNISTMLP import MNISTMLP

from models.utils.incremental_model import IncrementalModel


# function credit to https://github.com/facebookresearch/barlowtwins/blob/main/main.py
def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class RP2F(IncrementalModel):
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(RP2F, self).__init__(args)
        self.epochs = args.n_epochs
        self.net, self.net_old, self.classifier = None, None, None
        self.loss = F.cross_entropy
        self.net_arr = []

        self.device_ids = [0, 1]

        self.current_task = -1

    def begin_il(self, dataset):

        if self.args.dataset == 'seq-mnist':
            self.net = MNISTMLP(28 * 28, dataset.nc).to(self.device)
        else:
            if self.args.featureNet:
                self.net = MNISTMLP(1000, dataset.nc, hidden_dim=[800, 500]).to(self.device)
            elif self.args.backbone == 'None' or self.args.backbone == 'resnet18':
                self.net = resnet18(dataset.nc).to(self.device)
            elif self.args.backbone == 'resnet34':
                self.net = resnet34(dataset.nc).to(self.device)

        self.net.linear = None
        self.net.classifier = None

        self.net_old = copy.deepcopy(self.net)
        self.aux_net = copy.deepcopy(self.net)  # cal fisher for online model

        self.precision_matrices = {}
        for n, p in enumerate(self.net.parameters()):
            self.precision_matrices[n] = torch.zeros_like(p.data).to(self.device)

        self.latent_dim = self.net.nf * 8
        self.classifier = Classifier_Linear(self.latent_dim, dataset.nc).to(self.device)


        self.learner = copy.deepcopy(self.net).to(self.device)

        for n, p in enumerate(self.learner.parameters()):
            print(n, p.shape)
        for n, p in enumerate(self.learner.get_convs()):
            print(n, p.weight.shape)


        self.cpt = int(dataset.nc / dataset.nt)
        self.t_c_arr = dataset.t_c_arr
        self.eye = torch.tril(torch.ones((dataset.nc, dataset.nc))).bool().to(self.device)

    def train_task(self, dataset, train_loader):
        self.current_task += 1

        self.train_(train_loader)

        self.net_old = copy.deepcopy(self.net)
        # self.net_arr.append(
        #     copy.deepcopy(self.learner).to('cpu')
        # )

    def _diag_fisher(self, model, loader):
        cur_class = self.t_c_arr[self.current_task]

        if self.args.dataset == 'seq-cifar100':
            loader = DataLoader(loader.dataset,
                       batch_size=1024, shuffle=True, num_workers=0)

        precision_matrices = {}
        for n, p in enumerate(model.parameters()):
            # 计算非卷积层的权重设置为1
            if len(p.shape) == 1:
                precision_matrices[n] = torch.ones_like(p.data).to(self.device)
            else:
                precision_matrices[n] = torch.zeros_like(p.data).to(self.device)

        opt = torch.optim.SGD(model.parameters(), lr=self.args.lr)
        for conv_idx, conv in enumerate(model.get_convs()):
            conv.set_w_weight(copy.deepcopy(conv.weight.data))
            conv.weight.data = torch.ones_like(conv.weight.data).to(self.device) * self.args.eps_perturb


            conv_count, param_idx = -1, -1
            for n, p in enumerate(model.parameters()):
                if len(p.shape) != 1:
                    conv_count += 1
                if conv_count == conv_idx:
                    param_idx = n
                    break
            # print('connect:', conv_count, param_idx)

            sample_num = 0
            for step, data in enumerate(loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                sample_num += inputs.shape[0]

                feat = model.features(inputs)
                pred = self.classifier(feat)

                loss = self.loss(
                    pred[:, cur_class[0]: cur_class[-1] + 1],
                    labels - cur_class[0]
                )
                opt.zero_grad()
                loss.backward()

                for n, p in enumerate(model.parameters()):
                    if n == param_idx:
                        precision_matrices[n] += torch.abs(p.grad.data)
                        break

                if sample_num > 2000:
                    break

            precision_matrices[param_idx] = precision_matrices[param_idx] / sample_num + self.args.eta

            conv.weight.data = conv.w_weight
            conv.set_w_weight(None)

        return precision_matrices

    def train_(self, train_loader):
        cur_class = self.t_c_arr[self.current_task]
        print('learning classes: ', cur_class)

        cur_precision_matrices = {}
        for n, p in enumerate(self.net_old.parameters()):
            cur_precision_matrices[n] = torch.zeros_like(p.data).to(self.device)

        opt_learner = torch.optim.SGD(self.learner.parameters(), lr=self.args.lr)
        opt_classifier = torch.optim.SGD(self.classifier.parameters(), lr=self.args.clslr)

        scheduler_feature = StepLR(opt_learner, step_size=self.args.scheduler_step, gamma=0.1)
        scheduler_classifier = StepLR(opt_classifier, step_size=self.args.scheduler_step, gamma=0.1)
        for epoch in range(self.epochs):
            for step, data in enumerate(train_loader):
                inputs, labels = data[0].to(self.device), data[1].to(self.device)

                # =====  =====  ===== update classifier =====  =====  =====
                with torch.no_grad():
                    feat = self.net.features(inputs)

                pred = self.classifier(feat)
                loss_ce = self.loss(
                    pred[:, cur_class],
                    labels - cur_class[0]
                )
                classifier_loss = loss_ce

                opt_classifier.zero_grad()
                classifier_loss.backward()
                opt_classifier.step()

                # # =====  =====  ===== update target network =====  =====  =====
                for n, (online_params, target_params, old_params) \
                        in enumerate(zip(self.learner.parameters(), self.net.parameters(),
                                         self.net_old.parameters())):
                    online_weight, target_weight, old_weight = online_params.data, target_params.data, old_params.data
                    if self.current_task == 0:
                        target_params.data = online_weight * 1.
                    else:
                        cur_fisher, old_fisher = cur_precision_matrices[n], self.precision_matrices[n]
                        cur_fisher, old_fisher = cur_fisher / (cur_fisher + old_fisher), old_fisher / (
                                    cur_fisher + old_fisher)

                        target_params.data = old_fisher * old_weight + cur_fisher * online_weight

                # =====  =====  ===== update learner =====  =====  =====
                online_feat = self.learner.features(inputs)
                pred = self.classifier(online_feat)
                supervised_loss = self.loss(
                    pred[:, cur_class[0]: cur_class[-1] + 1],
                    labels - cur_class[0]
                )

                f_map = torch.transpose(online_feat, 0, 1)
                f_map = f_map - f_map.mean(dim=0, keepdim=True)
                f_map = f_map / torch.sqrt(self.args.eps + f_map.var(dim=0, keepdim=True))

                corr_mat = torch.matmul(f_map.t(), f_map)
                loss_mu = (off_diagonal(corr_mat).pow(2)).mean()

                learner_loss = supervised_loss + self.args.lambd * loss_mu

                opt_learner.zero_grad()
                learner_loss.backward(retain_graph=False)
                opt_learner.step()

            scheduler_feature.step()
            scheduler_classifier.step()
            if epoch % self.args.print_freq == 0:
                print('epoch:%d, feat_extract_loss:%.5f, classifier_loss:%.5f' % (
                    epoch, learner_loss.to('cpu').item(), classifier_loss.to('cpu').item()))

            for online_params, aux_params in zip(self.learner.parameters(), self.aux_net.parameters()):
                aux_params.data = online_params.data
            cur_precision_matrices = self._diag_fisher(self.aux_net, train_loader)

        for n, p in enumerate(self.net_old.parameters()):
            self.precision_matrices[n] += cur_precision_matrices[n]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cur_class = self.t_c_arr[self.current_task]
        # self.net.eval()
        x = x.to(self.device)
        with torch.no_grad():
            feat = self.net.features(x)
            outputs = self.classifier(feat)
            # outputs = outputs[:, :cur_class[-1] + 1]
        return outputs

    # def test_task(self, dataset, test_loader):
    #     self.net.to('cpu')
    #     dis = []
    #     for online_net in self.net_arr:
    #         dis_ = []
    #         for online_params, target_params \
    #                 in zip(online_net.parameters(), self.net.parameters()):
    #             online_weight, target_weight = online_params.data, target_params.data
    #             dis_.append(torch.mean((online_weight - target_weight) ** 2))
    #         dis.append(dis_)
    #     dis = torch.tensor(dis)
    #     # print(dis)
    #     print('avg_dis_per_layer:', torch.mean(dis, dim=0))
    #     print('avg_dis_per_net:', torch.mean(dis, dim=1))
    #     print('avg_dis:', torch.mean(dis))
    #
    #     self.net.to(self.device)


class Classifier_Linear(nn.Module):
    def __init__(self, dim, nc):
        super().__init__()
        self.linear = nn.Linear(dim, nc)

    def forward(self, x):
        y = self.linear(x)
        return y







def conv3x3(in_planes: int, out_planes: int, stride: int=1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return MyConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1)


class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int=1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.conv3 = MyConv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride)
            self.bn3 = nn.BatchNorm2d(self.expansion * planes)
            self.shortcut = nn.Sequential(
                self.conv3,
                self.bn3
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module):
    """
    ResNet network architecture. Designed for complex datasets.
    """

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.blocks = []
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = nn.Linear(nf * 8 * block.expansion, num_classes)

        self._features = nn.Sequential(self.conv1,
                                       self.bn1,
                                       self.layer1,
                                       self.layer2,
                                       self.layer3,
                                       self.layer4
                                       )
        self.classifier = self.linear

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block_ = block(self.in_planes, planes, stride)
            layers.append(block_)
            self.in_planes = planes * block.expansion
            self.blocks.append(block_)
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (output_classes)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)  # 64, 32, 32
        out = self.layer2(out)  # 128, 16, 16
        out = self.layer3(out)  # 256, 8, 8
        out = self.layer4(out)  # 512, 4, 4
        out = avg_pool2d(out, out.shape[2]) # 512, 1, 1
        out = out.view(out.size(0), -1)  # 512
        out = self.linear(out)
        return out

    def get_convs(self):
        conv_list = [self.conv1]
        for block in self.blocks:
            conv_list.append(block.conv1)
            conv_list.append(block.conv2)
            if hasattr(block, 'conv3'):
                conv_list.append(block.conv3)

        return conv_list

    def get_bns_params(self):
        params = []

        params += list(self.bn1.parameters())
        for block in self.blocks:
            params += list(block.bn1.parameters())
            params += list(block.bn2.parameters())
            if hasattr(block, 'bn3'):
                params += list(block.bn3.parameters())

        return params

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the non-activated output of the second-last layer.
        :param x: input tensor (batch_size, *input_shape)
        :return: output tensor (??)
        """
        out = self._features(x)
        out = avg_pool2d(out, out.shape[2])
        feat = out.view(out.size(0), -1)
        return feat

    def get_params(self) -> torch.Tensor:
        """
        Returns all the parameters concatenated in a single tensor.
        :return: parameters tensor (??)
        """
        params = []
        for pp in list(self.parameters()):
            params.append(pp.view(-1))
        return torch.cat(params)

    def set_params(self, new_params: torch.Tensor) -> None:
        """
        Sets the parameters to a given value.
        :param new_params: concatenated values to be set (??)
        """
        assert new_params.size() == self.get_params().size()
        progress = 0
        for pp in list(self.parameters()):
            cand_params = new_params[progress: progress +
                torch.tensor(pp.size()).prod()].view(pp.size())
            progress += torch.tensor(pp.size()).prod()
            pp.data = cand_params

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            grads.append(pp.grad.view(-1))
        return torch.cat(grads)


def resnet18(nclasses: int, nf: int=64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], nclasses, nf)

def resnet34(nclasses: int, nf: int=64) -> ResNet:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], nclasses, nf)

class MyConv2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups: int = 1,
                 padding_mode: str = 'zeros',
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MyConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        kernel_size_ = _pair(kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = Parameter(torch.empty(
            (out_channels, in_channels // groups, *kernel_size_), **factory_kwargs))

        self.w_weight = None # working memory weight

        self.register_parameter('bias', None)
        self.reset_parameters()

        self.f_map = None


    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def set_w_weight(self, w_weight):
        self.w_weight = w_weight

    def forward(self, input: Tensor) -> Tensor:

        if self.w_weight is None:
            weight = self.weight
        else:
            weight = self.weight + self.w_weight

        z = self._conv_forward(input, weight, self.bias)



        return z




