import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, inputs) -> Tensor:
        x, mask = inputs['x'], inputs['mask']
        identity = x

        out = self.conv1(torch.cat([x, mask], dim=1))
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')

        out += identity
        out = self.relu(out)

        return dict(x=out, mask=mask)


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock]],
        layers: List[int],
        num_classes: int = 101,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        face_parsing_classes=11,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        self.face_parsing_classes = face_parsing_classes
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512 * block.expansion, num_classes),
            nn.Softmax(1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    # type: ignore[arg-type]
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes+self.face_parsing_classes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes+self.face_parsing_classes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # import ipdb; ipdb.set_trace()
        mask = F.interpolate(mask, scale_factor=0.25, mode='nearest')
        x = self.layer1(dict(x=x, mask=mask))
        x = self.layer2(x)

        # mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        x = self.layer3(x)

        # mask = F.interpolate(mask, scale_factor=0.5, mode='nearest')
        x = self.layer4(x)['x']

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


def resnet18(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def Conv(in_channels, out_channels, kerner_size, stride, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kerner_size,
                  stride, padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


class TinyAge(nn.Module):
    def __init__(self, face_parsing_classes=11):
        super(TinyAge, self).__init__()
        self.conv1 = Conv(3+face_parsing_classes, 16, 3, 1, 1)
        self.conv2 = Conv(16, 16, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(16+face_parsing_classes, 32, 3, 1, 1)
        self.conv4 = Conv(32, 32, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(32+face_parsing_classes, 64, 3, 1, 1)
        self.conv6 = Conv(64, 64, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv7 = Conv(64+face_parsing_classes, 128, 3, 1, 1)
        self.conv8 = Conv(128, 128, 3, 1, 1)
        self.conv9 = Conv(128, 128, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.conv10 = Conv(128+face_parsing_classes, 128, 3, 1, 1)
        self.conv11 = Conv(128, 128, 3, 1, 1)
        self.conv12 = Conv(128, 128, 3, 1, 1)
        self.HP = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(128, 101),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        mask = mask.float()
        x = torch.cat([x, mask], dim=1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool1(x)
        x = torch.cat([x, mask], dim=1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = torch.cat([x, mask], dim=1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.pool3(x)
        x = torch.cat([x, mask], dim=1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.pool5(x)
        x = torch.cat([x, mask], dim=1)
        mask = F.interpolate(mask, scale_factor=0.5, mode='bilinear')
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.HP(x)
        x = x.view((x.size(0), -1))
        x = self.fc1(x.view((x.size(0), -1)))
        x = F.normalize(x, p=1, dim=1)

        return x


class ThinAge(nn.Module):
    def __init__(self, face_parsing_classes=11):
        super(ThinAge, self).__init__()
        self.conv1 = Conv(3, 32, 3, 1, 1)
        self.conv2 = Conv(32, 32, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv3 = Conv(32+face_parsing_classes, 64, 3, 1, 1)
        self.conv4 = Conv(64, 64, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv5 = Conv(64+face_parsing_classes, 128, 3, 1, 1)
        self.conv6 = Conv(128, 128, 3, 1, 1)
        self.conv7 = Conv(128, 128, 3, 1, 1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv8 = Conv(128+face_parsing_classes, 256, 3, 1, 1)
        self.conv9 = Conv(256, 256, 3, 1, 1)
        self.conv10 = Conv(256, 256, 3, 1, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv11 = Conv(256+face_parsing_classes, 256, 3, 1, 1)
        self.conv12 = Conv(256, 256, 3, 1, 1)
        self.conv13 = Conv(256, 256, 3, 1, 1)
        self.HP = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.AvgPool2d(kernel_size=7, stride=1)
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 101),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        x = self.conv1(x)
        x = self.conv2(x)
        # mask_1 = F.interpolate(mask.float(), scale_factor=0.5, mode='bilinear')
        x = self.pool1(x)
        mask_1 = F.interpolate(mask.float(), scale_factor=0.5, mode='bilinear')
        x = self.conv3(torch.cat([x, mask_1], dim=1))
        x = self.conv4(x)
        x = self.pool2(x)
        mask_2 = F.interpolate(
            mask_1.float(), scale_factor=0.5, mode='bilinear')
        x = self.conv5(torch.cat([x, mask_2], dim=1))
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        mask_3 = F.interpolate(
            mask_2.float(), scale_factor=0.5, mode='bilinear')
        x = self.conv8(torch.cat([x, mask_3], dim=1))
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.pool4(x)
        mask_4 = F.interpolate(
            mask_3.float(), scale_factor=0.5, mode='bilinear')
        x = self.conv11(torch.cat([x, mask_4], dim=1))
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.HP(x)
        x = x.view((x.size(0), -1))
        x = self.fc(x.view((x.size(0), -1)))
        x = F.normalize(x, p=1, dim=1)

        return x


if __name__ == '__main__':
    model = ThinAge()
    torch.save(model, './pretrained/ThinAge.pt')
