import torch
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['resnet18_sibnn_bireal_learn']


class GradientScale2D(nn.Module):
    def __init__(self, in_channels):
        super(GradientScale2D, self).__init__()
        # self.grad_scale = nn.Parameter(2*torch.ones(1, 1, 1), requires_grad=True)
        # self.grad_bias = nn.Parameter(0.4*torch.ones(1, 1, 1), requires_grad=True)
        self.grad_scale = nn.Parameter(torch.ones(in_channels, 1, 1), requires_grad=True)
        self.grad_bias = nn.Parameter(0.45*torch.zeros(in_channels, 1, 1), requires_grad=True)
    
    def forward(self, x):
        y = (x - self.grad_bias) * self.grad_scale
        return y

class AsymBinActiveF(torch.autograd.Function):
    '''
    Binarize the input activations to 0/1s
    '''
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = 0.5 * ((input).sign() + 1.0)
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1.0)] = 0
        grad_input[input.le(-0.3)] = 0
        return grad_input

class AsymBinActive(nn.Module):
    def __init__(self):
        super(AsymBinActive, self).__init__()
        self.binary_fn = AsymBinActiveF.apply

    def forward(self, x):
        x = self.binary_fn(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, input_channels, output_channels, 
            stride = -1, has_branch = True):
        super(BasicBlock, self).__init__()
        self.has_branch=has_branch

        self.bn1 = nn.BatchNorm2d(input_channels)
        self.gs1 = GradientScale2D(input_channels)
        self.act1 = AsymBinActive()
        self.conv1 = nn.Conv2d(input_channels, output_channels,
                            kernel_size=3, stride=stride, padding=1,
                            bias=False)
        self.bn2 = nn.BatchNorm2d(output_channels)

        self.bn3 = nn.BatchNorm2d(output_channels)
        self.gs2 = GradientScale2D(output_channels)
        self.act2 = AsymBinActive()
        self.conv2 = nn.Conv2d(output_channels, output_channels,
                            kernel_size=3, stride=1, padding=1,
                            bias=False)
        self.bn4 = nn.BatchNorm2d(output_channels)

        if has_branch:
            # self.branch_max = nn.Sequential(
            #     nn.MaxPool2d(kernel_size=2, stride=2),
            # )
            # self.branch_avg = nn.Sequential(
            #     nn.AvgPool2d(kernel_size=2, stride=2),
            # )

            self.fp32_downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size=1, \
                                 stride=1, bias=False),
                nn.BatchNorm2d(output_channels),
                nn.AvgPool2d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        # conv block
        ###### short-cut
        if self.has_branch:
            # short_cut_max = self.branch_max(x)
            # short_cut_avg = self.branch_avg(x)
            # short_cut = torch.cat([short_cut_max, short_cut_avg], 1)
            short_cut = self.fp32_downsample(x)
        else:
            short_cut = x
        ###### short-cut
        out = self.bn1(x)
        out = self.gs1(out)
        out = self.act1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out += short_cut

        short_cut = out
        out = self.bn3(out)
        out = self.gs2(out)
        out = self.act2(out)
        out = self.conv2(out)
        out = self.bn4(out) #HWGQ add another bn here
        out += short_cut

        return out


class ResNet18_SiBNN(nn.Module):

    def __init__(self, num_classes=1000):
        super(ResNet18_SiBNN, self).__init__()
        self.num_classes = num_classes
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            # nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            BasicBlock(64, 64, stride = 1, has_branch = False),
            BasicBlock(64, 64, stride = 1, has_branch = False),
            BasicBlock(64, 128, stride = 2, has_branch = True),
            BasicBlock(128, 128, stride = 1, has_branch = False),
            BasicBlock(128, 256, stride = 2, has_branch = True),
            BasicBlock(256, 256, stride = 1, has_branch = False),
            BasicBlock(256, 512, stride = 2, has_branch = True),
            BasicBlock(512, 512, stride = 1, has_branch = False)
        )
        self.relu=nn.ReLU(inplace=True)
        self.avgpool=nn.AvgPool2d(7, stride = 1)
        self.fc=nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x


def resnet18_sibnn_bireal_learn(pretrained=False, **kwargs):
    r"""ResNet18_SiBNN model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet18_SiBNN(**kwargs)
    if pretrained:
        model_path = 'model_list/resnet18_sibnn.pth.tar'
        pretrained_model = torch.load(model_path)
        model.load_state_dict(pretrained_model['state_dict'])
    return model

