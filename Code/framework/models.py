import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import framework.config as config


def move_data_to_gpu(x, device, using_float=False):
    # print(x.dtype)

    # At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported.
    x = x.copy()

    if using_float:
        x = torch.Tensor(x)
    else:
        if 'float' in str(x.dtype):
            x = torch.Tensor(x)

        elif 'int' in str(x.dtype):
            x = torch.LongTensor(x)

        else:
            raise Exception("Error!")

    x = x.to(device)

    return x

#####################################################################################################################

############################################################################################################
def init_layer(layer):
    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width

    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """Initialize a Batchnorm layer. """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

################################################################################

# ----------------------------------------------------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)):

        super(ConvBlock_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvBlock_dilation_single_layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2), padding=(1, 1)):

        super(ConvBlock_dilation_single_layer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=padding, bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_bn(self.bn1)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x




class ConvBlock_2nd_layer_dilation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), dilation=(2,2)):

        super(ConvBlock_2nd_layer_dilation, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=(int(kernel_size[0]/2), int(kernel_size[0]/2)), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=kernel_size, stride=(1, 1),
                               padding=(int(kernel_size[0]/2), int(kernel_size[0]/2)), bias=False, dilation=dilation)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weights()

    def init_weights(self):

        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'none':
            x = x
        else:
            raise Exception('Incorrect argument!')

        return x



class DR_BioL_CNN(nn.Module):
    def __init__(self, class_num, domain_num, dropout, MC_dropout, batchnormal=False):

        super(DR_BioL_CNN, self).__init__()

        self.MC_dropout = MC_dropout

        self.batchnormal = batchnormal
        if batchnormal:
            self.bn0 = nn.BatchNorm2d(config.mel_bins)

        self.conv_block1 = ConvBlock(config.MEL_feature_channel, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.dropout = dropout

        last_units = 512
        self.fc1 = nn.Linear(last_units, last_units, bias=True)

        # # ------------------- classification layer -----------------------------------------------------------------
        self.fc_final_event = nn.Linear(last_units, class_num, bias=True)

        location_dim = int(last_units / 2)
        self.fc_embedding_location = nn.Linear(last_units, location_dim, bias=True)
        self.fc_final_location = nn.Linear(location_dim, domain_num, bias=True)
        # ##############################################################################################################

    def mean_max(self, x):
        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2
        return x

    def forward(self, input):
        # print(input.shape) # torch.Size([32, 1, 100, 64])

        # (_, seq_len, mel_bins) = input.shape
        # x = input.view(-1, 1, seq_len, mel_bins)
        # '''(samples_num, feature_maps, time_steps, freq_num)'''

        x = input
        if self.batchnormal:
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)

        # -------------------------------------------------------------------------------------------------------------
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)

        x = torch.mean(x, dim=3)

        (x1, _) = torch.max(x, dim=2)
        x2 = torch.mean(x, dim=2)
        x = x1 + x2

        encoder  = x
        common_embeddings = F.relu_(self.fc1(x))
 
        location_embeddings = F.gelu(self.fc_embedding_location(x))
        # -------------------------------------------------------------------------------------------------------------

        if self.MC_dropout:
            event = self.fc_final_event(F.dropout(common_embeddings, p=self.dropout))
            location = self.fc_final_location(F.dropout(location_embeddings, p=self.dropout))
        else:
            event = self.fc_final_event(F.dropout(common_embeddings, p=self.dropout, training=self.training))
            location = self.fc_final_location(F.dropout(location_embeddings, p=self.dropout, training=self.training))

        return event, location, common_embeddings, location_embeddings




