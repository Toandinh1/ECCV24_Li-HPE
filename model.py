

import torch
import torch.nn as nn
from torchvision.transforms import Resize
import torch.nn.functional as F





class regression(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super(regression, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim*2)
        self.fc3 = nn.Linear(hidden_dim*2, output_dim)
        self.bn = nn.BatchNorm1d(hidden_dim*2)
        self.relu = nn.ReLU()  # Hàm kích hoạt ReLU
        self.dropout = nn.Dropout(p=0.1) 
        self.gelu = nn.GELU()
    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        #x= x.reshape(x.size(0), x.size(1)*x.size(2))
        
        x = self.fc1(x)
        x = self.relu(x)  # Áp dụng ReLU sau lớp fully connected thứ nhất
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn(x)
        x = self.relu(x)  # Áp dụng ReLU sau lớp fully connected thứ hai
        x = self.dropout(x)
        output = self.fc3(x)

        return output
 

class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(Conv2D, self).__init__()
        self.conv2d_1 = nn.Conv2d(in_channels, out_channels, kernel_size = 9, padding = 5, bias=bias)
        self.conv2d_2 = nn.Conv2d(in_channels, out_channels, kernel_size = 7, padding = 4,  bias=bias)
        self.conv2d_3 = nn.Conv2d(in_channels, out_channels, kernel_size = 5, padding = 3, bias=bias)
        self.conv2d_4 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 2, bias=bias)
        self.relu = nn.ReLU()
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.pool = torch.nn.MaxPool2d(kernel_size=2)
        self.dropout = nn.Dropout(p=0.1) 
        
        
    def forward(self, x):
       
        
        x1 = self.conv2d_1(x)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.batch_norm(x1)
        x2 = self.conv2d_2(x)
        x2 = self.relu(x2)
        x2 = self.dropout(x2)
        x2 = self.batch_norm(x2)
        x3 = self.conv2d_3(x)
        x3 = self.relu(x3)
        x3 = self.dropout(x3)
        x3 = self.batch_norm(x3)
        x4 = self.conv2d_4(x)
        x4 = self.relu(x4)
        x4 = self.dropout(x4)
        x4 = self.batch_norm(x4)
        x_out = torch.cat([x1,x2,x3,x4], dim=1)
        #x_out = self.pool(x_out)
        return x_out

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)  # Sử dụng 2 * hidden_dim do là BiLSTM
        self.dropout = nn.Dropout(p=0.1)
    def forward(self, x):
        # Đảm bảo rằng kích thước đầu vào phù hợp (32, 3, 114, 10) -> (32, 3, 1140)
        x = x.view(x.size(0), x.size(1), -1)

        # Truyền qua mạng BiLSTM
        out, _ = self.bilstm(x)

        # Đưa qua mạng fully connected để thu được đầu ra mong muốn (32, 32, 17, 2)
        out = self.fc(out)
        out = self.dropout(out)
        
        return out



    
class Attention(nn.Module):
    def __init__(self, input_dim, return_sequences=True):
        super(Attention, self).__init__()
        self.return_sequences = return_sequences
        self.W = nn.Parameter(torch.randn(input_dim, 1))
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        e = torch.tanh(torch.matmul(x, self.W) + self.b)
        a = F.softmax(e, dim=1)
        output = x * a

        if self.return_sequences:
            return output

        return torch.sum(output, dim=1)


    


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, num_blocks=4):
        super(ResNet, self).__init__()

        self.in_channels = 32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(ResidualBlock, 32, num_blocks, stride=1)
        self.layer2 = self._make_layer(ResidualBlock, 64, num_blocks, stride=1)
        self.layer3 = self._make_layer(ResidualBlock, 128, num_blocks, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((18, 2))
        self.decode = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        batch = x.shape[0]
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        #out = self.decode(out)
        #out = self.avg_pool(out)
        
        #m = torch.nn.AvgPool2d((3, 3), stride=(2, 2))
        #out = m(out)
        #out = out.view(batch,18,2)
        #out = self.fc(out)

        return out





class WiPose_benchmark(nn.Module):
    def __init__(self):
        super(WiPose_benchmark, self).__init__()

        # 4 CNN layers without max pooling
        self.cnn1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.cnn2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()

        self.cnn3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()

        self.cnn4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.relu4 = nn.ReLU()

        # LSTM layers
        self.lstm1 = nn.LSTM(input_size=450, hidden_size=256, num_layers=1, batch_first=True, dropout=0.5)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, batch_first=True, dropout=0.5)
        self.lstm3 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, batch_first=True, dropout=0.5)

        # Fully connected layer for final output
        self.fc = nn.Linear(64, 2 * 18)

    def forward(self, x):
        # CNN layers
        x = self.relu1(self.bn1(self.cnn1(x)))
        x = self.relu2(self.bn2(self.cnn2(x)))
        x = self.relu3(self.bn3(self.cnn3(x)))
        x = self.relu4(self.bn4(self.cnn4(x)))

        # Reshape for LSTM input
        x = x.view(x.size(0), x.size(1), x.size(2)*x.size(3))

        # LSTM layers
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)

        # Fully connected layer
        #x = self.fc(x[:, -1, :])  # Take the output of the last time step

        # Reshape to match the desired output size
        #x = x.view(-1, 2, 18)

        return x
    
class ResidualBlock_WiMose(nn.Module):
    expansion = 1  # Thêm thuộc tính expansion

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock_WiMose, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out += residual
        out = self.relu3(out)
        return out

class WiMose(nn.Module):
    def __init__(self, block, layers):
        super(WiMose, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=1)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=1)
        

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
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

        
        return x
