import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 16
        super(ResNet, self).__init__()
        self.conv1 = conv3x3(3, 16)
        #, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, padding=1)
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 128, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(3)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.lastFC = nn.Linear(128, 128)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

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
        #print 'X shape before flatten:', x.shape
        x = x.view(x.size(0), -1)
        #print 'X shape after flatten:', x.shape
        x = self.lastFC(x)
        #x = self.fc(x)
        return x


# In[27]:


# Feature Net
class BaseSimpleFeatureNet(nn.Module):
    def __init__(self):
        super(BaseSimpleFeatureNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding = 4)
        self.maxpool1 = nn.MaxPool2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.maxpool2 = nn.MaxPool2d(32)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.maxpool3 = nn.MaxPool2d(64)
        self.fc1 = nn.Linear(128, 128)
        #self.fc2 = nn.Linear(1024, 128)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = F.relu(x)
        x = self.conv2(x)
        #x = self.maxpool2(x)
        x = F.relu(x)
        #x = self.conv3(x)
        #x = self.maxpool3(x)
        #x = F.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
        #x = F.relu(x)
        #x = self.fc2(x)
        #return x


# Lighting Net
class LightingNet(nn.Module):
    def __init__(self):
        super(LightingNet, self).__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 27) #18)

    def forward(self, x):
        #print x.shape
        x = self.fc1(x)
        x = F.relu(x)
        #print x.shape
        x = F.dropout(x)
        x = self.fc2(x)
        return x;


# In[29]:


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = nn.Linear(128, 128)
        self.fc1 = nn.Linear(128, 1)
        self.dOut = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        x = self.dOut(x)
        x = self.fc(x)
        x = F.relu(x)
        x = self.dOut(x)
        x = self.fc1(x)
        x = F.tanh(x)
        return x
