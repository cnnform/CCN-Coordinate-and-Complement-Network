# encoding:utf-8
# Modify from torchvision
# ResNeXt: Copy from https://github.com/last-one/tools/blob/master/pytorch/SE-ResNeXt/SeResNeXt.py
import torch
import torch.nn as nn
import math



class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, hw,stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.planes = planes
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn4 = nn.BatchNorm2d(planes * 4)
        self.bn5 = nn.BatchNorm2d(planes * 4)
        self.bn6 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        # CHW
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_down = nn.Conv2d(planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(planes // 4, planes * 4, kernel_size=1, bias=False)
        self.conv_dowm_1 = nn.Conv2d(hw , hw//4, kernel_size=1,bias=False)
        self.conv_up_1 = nn.Conv2d(hw//4 , hw, kernel_size=1,bias=False)
        self.conv_dowm_2 = nn.Conv2d(hw , hw//4, kernel_size=1,bias=False)
        self.conv_up_2 = nn.Conv2d(hw//4 , hw, kernel_size=1,bias=False)
        self.sig = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1, planes*4, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1,1,hw,1))
        self.beta = nn.Parameter(torch.ones(1,1,1,hw))
        # #fusion
        self.abc = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

        # Downsample
        self.downsample = downsample
        self.stride = stride


    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # c
        out6 = self.global_pool(out)
        out6 = self.conv_down(out6)
        out6 = self.relu(out6)
        out6 = self.conv_up(out6)
        out6 = self.sig(out6)
        out6 = self.bn4(self.alpha *out6 * out)

        # h
        out1 = out6.permute(0,2,1,3)
        out1 =self.global_pool(out1)
        out1 = self.conv_dowm_1(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up_1(out1)
        out1 = self.sig(out1)
        out1 = out1.permute(0,2,1,3)
        out1 = self.bn5(self.gamma * out1 *out6)


        # w
        out3 = out1.permute(0,3,2,1)
        out3 =self.global_pool(out3)
        out3 = self.conv_dowm_2(out3)
        out3 = self.relu(out3)
        out3 = self.conv_up_2(out3)
        out3 = self.sig(out3)
        out3 = out3.permute(0,3,2,1)
        out3 =torch.tanh(self.beta*out3 * out1)

        energy = out3+out
        attention = self.softmax(energy)
        out4 = out*attention
        out = self.abc*out4+out3

        if self.downsample is not None:
            residual = self.downsample(x)
        out = out +residual
        res = self.relu(out)
        return res



class CCNResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(CCNResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, 56 , layers[0])
        self.layer2 = self._make_layer(block, 128, 28  ,layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 14  ,layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512,  7 ,layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes,hw, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes,hw, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,hw))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x



class BottleneckX(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, hw,cardinality, stride=1, downsample=None):
        super(BottleneckX, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes * 2, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes * 2)

        self.conv2 = nn.Conv2d(planes * 2, planes * 2, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(planes * 2)

        self.conv3 = nn.Conv2d(planes * 2, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.bn4 = nn.BatchNorm2d(planes * 4)
        self.bn5 = nn.BatchNorm2d(planes * 4)
        self.bn6 = nn.BatchNorm2d(planes * 4)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool_1 = nn.AdaptiveAvgPool2d((1,None))
        self.global_pool_2 = nn.AdaptiveAvgPool2d((None,1))
        self.conv_down = nn.Conv2d(
            planes * 4, planes // 4, kernel_size=1, bias=False)
        self.conv_up = nn.Conv2d(
            planes // 4, planes * 4, kernel_size=1, bias=False)
        self.conv_dowm_1 = nn.Conv2d(hw , hw//4, kernel_size=1,bias=False)
        self.conv_up_1 = nn.Conv2d(hw//4 , hw, kernel_size=1,bias=False)
        self.sig = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1, planes*4, 1, 1))
        self.gamma = nn.Parameter(torch.ones(1,1,hw,1))
        self.beta = nn.Parameter(torch.ones(1,1,1,hw))
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.abc = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        # c
        out6 = self.global_pool(out)
        out6 = self.conv_down(out6)
        out6 = self.relu(out6)
        out6 = self.conv_up(out6)
        out6 = self.sig(out6)
        out6 = self.bn4(self.alpha *out6 * out)

        # h
        out1 = out6.permute(0,2,1,3)
        out1 =self.global_pool(out1)
        out1 = self.conv_dowm_1(out1)
        out1 = self.relu(out1)
        out1 = self.conv_up_1(out1)
        out1 = self.sig(out1)
        out1 = out1.permute(0,2,1,3)
        out1 = self.bn5(self.gamma * out1 *out6)


        # w
        out3 = out1.permute(0,3,2,1)
        out3 =self.global_pool(out3)
        out3 = self.conv_dowm_1(out3)
        out3 = self.relu(out3)
        out3 = self.conv_up_1(out3)
        out3 = self.sig(out3)
        out3 = out3.permute(0,3,2,1)
        out3 =torch.tanh(self.beta*out3 * out1)


        energy = out3+out
        attention = self.softmax(energy)
        out4 = out*attention
        out = self.abc*out4+out3

        if self.downsample is not None:
            residual = self.downsample(x)

        res = out + residual
        res = self.relu(res)

        return res


class CCNResNeXt(nn.Module):

    def __init__(self, block, layers, cardinality=32, num_classes=1000):
        super(CCNResNeXt, self).__init__()
        self.cardinality = cardinality
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, 56,layers[0])
        self.layer2 = self._make_layer(block, 128,28, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, 14,layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512,7, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes,hw, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, hw,self.cardinality, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, hw,self.cardinality))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x


def CCN_resnet50(**kwargs):
    """Constructs a CCN-ResNet-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = CCNResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def CCN_resnet101(**kwargs):
    """Constructs a CCN-ResNet-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = CCNResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def CCN_resnet152(**kwargs):
    """Constructs a CCN-ResNet-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = CCNResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def CCN_resnext50(**kwargs):
    """Constructs a CCN-ResNeXt-50 model.
    Args:
        num_classes = 1000 (default)
    """
    model = CCNResNeXt(BottleneckX, [3, 4, 6, 3], **kwargs)
    return model


def CCN_resnext101(**kwargs):
    """Constructs a CCN-ResNeXt-101 model.
    Args:
        num_classes = 1000 (default)
    """
    model = CCNResNeXt(BottleneckX, [3, 4, 23, 3], **kwargs)
    return model


def CCN_resnext152(**kwargs):
    """Constructs a CCN-ResNeXt-152 model.
    Args:
        num_classes = 1000 (default)
    """
    model = CCNResNeXt(BottleneckX, [3, 8, 36, 3], **kwargs)
    return model
