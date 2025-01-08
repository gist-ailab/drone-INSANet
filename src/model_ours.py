from math import sqrt
from itertools import product as product
import warnings

import torch
from torch import nn
import torch.nn.functional as F
import torchvision

from insa import *
from attention import *
from utils.utils import *

from model import INSANet, PredictionConvolutions
warnings.filterwarnings(action='ignore')

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VGGEdited(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self,device,  attention =['CBAM', 'CHANNEL', 'SPATIAL'], pos = ['first', 'last', 'all'], fusion = ['add', 'cat']):
        super(VGGEdited, self).__init__()
        self.attention = attention
        self.pos = pos
        self.fusion = fusion
        self.device = device
        # RGB
        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)        
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)

        # LWIR
        self.conv1_1_lwir = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_lwir = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_lwir = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_lwir = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_lwir = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_lwir = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_lwir = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_lwir = nn.BatchNorm2d(128, affine=True)

        self.pool2_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_lwir = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_lwir = nn.BatchNorm2d(256, affine=True)
        
        # weight-sharing network
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6_1_bn = nn.BatchNorm2d(512, affine=True)  
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1)
        
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.conv7_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv7_2_bn = nn.BatchNorm2d(512, affine=True)  
        
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        nn.init.xavier_uniform_(self.conv8_1.weight)
        nn.init.constant_(self.conv8_1.bias, 0.)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv8_2.weight)
        nn.init.constant_(self.conv8_2.bias, 0.)

        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.xavier_uniform_(self.conv9_1.weight)
        nn.init.constant_(self.conv9_1.bias, 0.)
        self.conv9_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv9_2.weight)
        nn.init.constant_(self.conv9_2.bias, 0.)

        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.xavier_uniform_(self.conv10_1.weight)
        nn.init.constant_(self.conv10_1.bias, 0.)
        self.conv10_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv10_2.weight)
        nn.init.constant_(self.conv10_2.bias, 0.)

        self.conv1x1_vis = nn.Conv2d(256,256,kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1x1_vis.weight.data.normal_(0, 0.01)
        self.conv1x1_vis.bias.data.fill_(0.01)

        self.conv1x1_lwir = nn.Conv2d(256,256,kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1x1_lwir.weight.data.normal_(0, 0.01)
        self.conv1x1_lwir.bias.data.fill_(0.01)   
        
        self.weight = 0.5
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  
        nn.init.constant_(self.rescale_factors, 20)

        # Load pretrained layers
        self.load_pretrained_layers()

        # # INtra-INter Attention (INSA) module
        self.insa = INSA(n_iter=2,
                         dim=256,
                         n_head=1,
                         ffn_dim=4)

        self.pre_att_rgb = self.GetPosition()
        self.pre_att_ir = self.GetPosition()
        self.post_att, self.post_channel = self.GetPostAttention(256)
        if fusion == 'cat':
            self.reduction_conv = nn.Conv2d(self.post_channel, 256, kernel_size=1, padding=0, stride=1, bias=True)

    def GetPosition(self):
        att = []
        if self.pos == 'first':
            channel = 64
            att.append(self.GetPreAttention(channel))
        elif self.pos == 'last':
            channel = 256
            att.append(self.GetPreAttention(channel))
        elif self.pos == 'all':
            att.append(self.GetPreAttention(64))
            att.append(self.GetPreAttention(128))
            att.append(self.GetPreAttention(256))
        return att

    def GetPreAttention(self, channel):
        if self.attention == 'CBAM':
            return CBAM(channel)
        elif self.attention == 'CHANNEL':
            return ChannelGate(channel)
        elif self.attention == 'SPATIAL':
            return SpatialGate()
        
    def GetPostAttention(self, channel):
        if self.fusion == 'add':
            channel = channel
        elif self.fusion == 'cat':
            if self.pos == 'last':
                channel = 256
            elif self.pos == 'all':
                channel = 64+128+256
            channel = channel * 2
        if self.attention == 'CBAM':
            att= CBAM(channel).to("cuda")
        elif self.attention == 'CHANNEL':
            att= ChannelGate(channel).to("cuda")
        elif self.attention == 'SPATIAL':
            att=SpatialGate().to("cuda")
            channel = channel
        else:
            print(self.fusion)
        return att, channel

    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        # RGB
        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis)))  
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis))) 
        out_vis = self.pool1_vis(out_vis) 
        if self.pos == 'first' or self.pos == 'all':
            self.pre_att_rgb[0] = self.pre_att_rgb[0].to(out_vis.device)
            out_vis_1 = self.pre_att_rgb[0](out_vis)
            out_vis_2 = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis_1)))
        else:
            out_vis_2 = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))

        out_vis_2 = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis_2))) 
        out_vis_2 = self.pool2_vis(out_vis_2) 

        if  self.pos == 'all':
            self.pre_att_rgb[1] = self.pre_att_rgb[1].to(out_vis_2.device)
            out_vis_2 = self.pre_att_rgb[1](out_vis_2)
            out_vis_3 = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis_2))) 
        else:
            out_vis_3 = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis_2))) 

        # out_vis = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis_2))) 
        out_vis_3 = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis_3))) 
        out_vis_3 = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis_3)))
        # if self.pos == 'last' or self.pos == 'all':
        #     out_vis_3 = self.pre_att_rgb(out_vis_3)
        if self.pos == 'last' or self.pos == 'all':
            self.pre_att_rgb[-1] = self.pre_att_rgb[-1].to(out_vis_3.device)
            out_vis_3 = self.pre_att_rgb[-1](out_vis_3)
        out_vis = out_vis_3

        # LWIR
        out_lwir_1 = F.relu(self.conv1_1_bn_lwir(self.conv1_1_lwir(image_lwir)))  
        out_lwir_1 = F.relu(self.conv1_2_bn_lwir(self.conv1_2_lwir(out_lwir_1))) 
        out_lwir_1 = self.pool1_lwir(out_lwir_1)
        if self.pos == 'first' or self.pos == 'all':
            self.pre_att_ir[0] = self.pre_att_ir[0].to(out_lwir_1.device)
            out_lwir_1 = self.pre_att_ir[0](out_lwir_1)
            out_lwir_2 = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir_1)))
        else:
            out_lwir_2 = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir_1)))
        out_lwir_2 = F.relu(self.conv2_2_bn_lwir(self.conv2_2_lwir(out_lwir_2))) 
        out_lwir_2 = self.pool2_lwir(out_lwir_2) 
        if self.pos == 'all':
            self.pre_att_ir[1] = self.pre_att_ir[1].to(out_lwir_2.device)
            out_lwir_2 = self.pre_att_ir[1](out_lwir_2)
            out_lwir_3 = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir_2)))
        else:
            out_lwir_3 = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir_2))) 
        out_lwir_3 = F.relu(self.conv3_2_bn_lwir(self.conv3_2_lwir(out_lwir_3))) 
        out_lwir_3 = F.relu(self.conv3_3_bn_lwir(self.conv3_3_lwir(out_lwir_3))) 
        if self.pos == 'last' or self.pos == 'all':
            self.pre_att_ir[-1] = self.pre_att_ir[-1].to(out_lwir_3.device)
            out_lwir_3 = self.pre_att_ir[-1](out_lwir_3)
            out_lwir_3 = out_lwir_3

        out_vis = F.relu(self.conv1x1_vis(out_vis_3))
        out_lwir = F.relu(self.conv1x1_lwir(out_lwir_3))


        out_vis, out_lwir = self.insa(out_vis, out_lwir)

        if self.fusion == 'cat':
            if self.pos == 'all':
                #match all sizees into layer3
                out_vis_1 = F.interpolate(out_vis_1, size=(out_vis.size(2), out_vis.size(3)), mode='bilinear')
                out_lwir_1 = F.interpolate(out_lwir_1, size=(out_lwir.size(2), out_lwir.size(3)), mode='bilinear')
                out_vis_2 = F.interpolate(out_vis_2, size=(out_vis.size(2), out_vis.size(3)), mode='bilinear')
                out_lwir_2 = F.interpolate(out_lwir_2, size=(out_lwir.size(2), out_lwir.size(3)), mode='bilinear')
                out = torch.cat((out_vis_1, out_lwir_1, out_vis_2, out_lwir_2, out_vis_3, out_lwir_3), 1)
                out = self.post_att(out)
                out = self.reduction_conv(out)
            elif self.pos == 'last':
                out = torch.cat((out_vis, out_lwir), 1)
                out = self.post_att(out)
                out = self.reduction_conv(out)
        
        elif self.fusion == 'add':
            out = out_vis + out_lwir
            out = self.post_att(out)
        else:
            out = torch.add(out_vis * self.weight, out_lwir *(self.weight) )
        # weight-sharing network
        out = self.pool3(out)

        out = F.relu(self.conv4_1_bn(self.conv4_1(out))) 
        out = F.relu(self.conv4_2_bn(self.conv4_2(out))) 
        out = F.relu(self.conv4_3_bn(self.conv4_3(out))) 
        out = self.pool4(out)
        conv4_3_feats = out
        
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        out = F.relu(self.conv5_1_bn(self.conv5_1(out))) 
        out = F.relu(self.conv5_2_bn(self.conv5_2(out))) 
        out = F.relu(self.conv5_3_bn(self.conv5_3(out))) 
        out = self.pool5(out)
        
        out = F.relu(self.conv6_1_bn(self.conv6_1(out)))
        out = F.relu(self.conv6_2(out))
        conv6_feats = out

        out = F.relu(self.conv7_1(out))
        out = F.relu(self.conv7_2_bn(self.conv7_2(out)))
        conv7_feats = out

        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out)) 
        conv8_feats = out  

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_feats = out 

        out = F.relu(self.conv10_1(out)) 
        out = F.relu(self.conv10_2(out)) 
        conv10_feats = out
        
        return conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats

    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of model
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG_BN
        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[1:50]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[50:99]):    
            if param == 'conv1_1_lwir.weight':
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]][:, :1, :, :]              
            else:
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        self.load_state_dict(state_dict)

        print("Load Model: AttNet\n")


class VGGATTINSA(nn.Module):
    """
    VGG base convolutions to produce lower-level feature maps.
    """
    def __init__(self,device,  attention =['CBAM', 'CHANNEL', 'SPATIAL'], pos = ['first', 'last', 'all'], fusion = ['add', 'cat']):
        super(VGGATTINSA, self).__init__()
        self.attention = attention
        self.pos = pos
        self.fusion = fusion
        self.device = device
        # RGB
        self.conv1_1_vis = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_vis = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_vis = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_vis = nn.BatchNorm2d(64, affine=True)        
        self.pool1_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_vis = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_vis = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_vis = nn.BatchNorm2d(128, affine=True)
        self.pool2_vis = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_vis = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_vis = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_vis = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_vis = nn.BatchNorm2d(256, affine=True)

        # LWIR
        self.conv1_1_lwir = nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=True) 
        self.conv1_1_bn_lwir = nn.BatchNorm2d(64, affine=True)
        self.conv1_2_lwir = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=True)
        self.conv1_2_bn_lwir = nn.BatchNorm2d(64, affine=True)
        
        self.pool1_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2_1_lwir = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_1_bn_lwir = nn.BatchNorm2d(128, affine=True)
        self.conv2_2_lwir = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.conv2_2_bn_lwir = nn.BatchNorm2d(128, affine=True)

        self.pool2_lwir = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3_1_lwir = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_1_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_2_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_2_bn_lwir = nn.BatchNorm2d(256, affine=True)
        self.conv3_3_lwir = nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=True)
        self.conv3_3_bn_lwir = nn.BatchNorm2d(256, affine=True)
        
        # weight-sharing network
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 
        
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv4_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool4 = nn.MaxPool2d(kernel_size=3,padding=1,stride=1, ceil_mode=True)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_1_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_2_bn = nn.BatchNorm2d(512, affine=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv5_3_bn = nn.BatchNorm2d(512, affine=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True) 

        self.conv6_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=True)
        self.conv6_1_bn = nn.BatchNorm2d(512, affine=True)  
        self.conv6_2 = nn.Conv2d(512, 512, kernel_size=1)
        
        self.conv7_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        self.conv7_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        self.conv7_2_bn = nn.BatchNorm2d(512, affine=True)  
        
        self.conv8_1 = nn.Conv2d(512, 256, kernel_size=1, stride=2)
        nn.init.xavier_uniform_(self.conv8_1.weight)
        nn.init.constant_(self.conv8_1.bias, 0.)
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv8_2.weight)
        nn.init.constant_(self.conv8_2.bias, 0.)

        self.conv9_1 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.xavier_uniform_(self.conv9_1.weight)
        nn.init.constant_(self.conv9_1.bias, 0.)
        self.conv9_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv9_2.weight)
        nn.init.constant_(self.conv9_2.bias, 0.)

        self.conv10_1 = nn.Conv2d(512, 256, kernel_size=1)
        nn.init.xavier_uniform_(self.conv10_1.weight)
        nn.init.constant_(self.conv10_1.bias, 0.)
        self.conv10_2 = nn.Conv2d(256, 512, kernel_size=3, padding=1, bias=True)
        nn.init.xavier_uniform_(self.conv10_2.weight)
        nn.init.constant_(self.conv10_2.bias, 0.)

        self.conv1x1_vis = nn.Conv2d(256,256,kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1x1_vis.weight.data.normal_(0, 0.01)
        self.conv1x1_vis.bias.data.fill_(0.01)

        self.conv1x1_lwir = nn.Conv2d(256,256,kernel_size=1, padding=0, stride=1, bias=True)
        self.conv1x1_lwir.weight.data.normal_(0, 0.01)
        self.conv1x1_lwir.bias.data.fill_(0.01)   
        
        self.weight = 0.5
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  
        nn.init.constant_(self.rescale_factors, 20)

        # Load pretrained layers
        self.load_pretrained_layers()

        # # INtra-INter Attention (INSA) module
        # self.insa = INSA(n_iter=2,
        #                  dim=256,
        #                  n_head=1,
        #                  ffn_dim=4)

        self.pre_att_rgb = self.GetPosition()
        self.pre_att_ir = self.GetPosition()
        self.post_att, self.post_channel = self.GetPostAttention(256)
        if fusion == 'cat':
            self.reduction_conv = nn.Conv2d(self.post_channel, 256, kernel_size=1, padding=0, stride=1, bias=True)

    def GetPosition(self):
        att = []
        if self.pos == 'first':
            channel = 64
            att.append(self.GetPreAttention(channel))
        elif self.pos == 'last':
            channel = 256
            att.append(self.GetPreAttention(channel))
        elif self.pos == 'all':
            att.append(self.GetPreAttention(64))
            att.append(self.GetPreAttention(128))
            att.append(self.GetPreAttention(256))
        return att

    def GetPreAttention(self, channel):
        if self.attention == 'CBAM':
            return CBAM(channel)
        elif self.attention == 'CHANNEL':
            return ChannelGate(channel)
        elif self.attention == 'SPATIAL':
            return SpatialGate()
        
    def GetPostAttention(self, channel):
        if self.fusion == 'add':
            channel = channel
        elif self.fusion == 'cat':
            if self.pos == 'last':
                channel = 256
            elif self.pos == 'all':
                channel = 64+128+256
            channel = channel * 2
        if self.attention == 'CBAM':
            att= CBAM(channel).to("cuda")
        elif self.attention == 'CHANNEL':
            att= ChannelGate(channel).to("cuda")
        elif self.attention == 'SPATIAL':
            att=SpatialGate().to("cuda")
            channel = channel
        else:
            print(self.fusion)
        return att, channel

        

    def forward(self, image_vis, image_lwir):
        """
        Forward propagation.

        :param image: images, a tensor of dimensions (N, 3, 300, 300)
        :return: lower-level feature maps conv4_3 and conv7
        """
        # RGB
        out_vis = F.relu(self.conv1_1_bn_vis(self.conv1_1_vis(image_vis)))  
        out_vis = F.relu(self.conv1_2_bn_vis(self.conv1_2_vis(out_vis))) 
        out_vis = self.pool1_vis(out_vis) 
        if self.pos == 'first' or self.pos == 'all':
            self.pre_att_rgb[0] = self.pre_att_rgb[0].to(out_vis.device)
            out_vis_1 = self.pre_att_rgb[0](out_vis)
            out_vis_2 = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis_1)))
        else:
            out_vis_2 = F.relu(self.conv2_1_bn_vis(self.conv2_1_vis(out_vis)))

        out_vis_2 = F.relu(self.conv2_2_bn_vis(self.conv2_2_vis(out_vis_2))) 
        out_vis_2 = self.pool2_vis(out_vis_2) 

        if  self.pos == 'all':
            self.pre_att_rgb[1] = self.pre_att_rgb[1].to(out_vis_2.device)
            out_vis_2 = self.pre_att_rgb[1](out_vis_2)
            out_vis_3 = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis_2))) 
        else:
            out_vis_3 = F.relu(self.conv3_1_bn_vis(self.conv3_1_vis(out_vis_2))) 

        out_vis_3 = F.relu(self.conv3_2_bn_vis(self.conv3_2_vis(out_vis_3))) 
        out_vis_3 = F.relu(self.conv3_3_bn_vis(self.conv3_3_vis(out_vis_3)))
        if self.pos == 'last' or self.pos == 'all':
            self.pre_att_rgb[-1] = self.pre_att_rgb[-1].to(out_vis_3.device)
            out_vis_3 = self.pre_att_rgb[-1](out_vis_3)
        out_vis = out_vis_3

        # LWIR
        out_lwir_1 = F.relu(self.conv1_1_bn_lwir(self.conv1_1_lwir(image_lwir)))  
        out_lwir_1 = F.relu(self.conv1_2_bn_lwir(self.conv1_2_lwir(out_lwir_1))) 
        out_lwir_1 = self.pool1_lwir(out_lwir_1)
        if self.pos == 'first' or self.pos == 'all':
            self.pre_att_ir[0] = self.pre_att_ir[0].to(out_lwir_1.device)
            out_lwir_1 = self.pre_att_ir[0](out_lwir_1)
            out_lwir_2 = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir_1)))
        else:
            out_lwir_2 = F.relu(self.conv2_1_bn_lwir(self.conv2_1_lwir(out_lwir_1)))
        out_lwir_2 = F.relu(self.conv2_2_bn_lwir(self.conv2_2_lwir(out_lwir_2))) 
        out_lwir_2 = self.pool2_lwir(out_lwir_2) 
        if self.pos == 'all':
            self.pre_att_ir[1] = self.pre_att_ir[1].to(out_lwir_2.device)
            out_lwir_2 = self.pre_att_ir[1](out_lwir_2)
            out_lwir_3 = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir_2)))
        else:
            out_lwir_3 = F.relu(self.conv3_1_bn_lwir(self.conv3_1_lwir(out_lwir_2))) 
        out_lwir_3 = F.relu(self.conv3_2_bn_lwir(self.conv3_2_lwir(out_lwir_3))) 
        out_lwir_3 = F.relu(self.conv3_3_bn_lwir(self.conv3_3_lwir(out_lwir_3))) 
        if self.pos == 'last' or self.pos == 'all':
            self.pre_att_ir[-1] = self.pre_att_ir[-1].to(out_lwir_3.device)
            out_lwir_3 = self.pre_att_ir[-1](out_lwir_3)
            out_lwir_3 = out_lwir_3

        out_vis = F.relu(self.conv1x1_vis(out_vis_3))
        out_lwir = F.relu(self.conv1x1_lwir(out_lwir_3))

        if self.fusion == 'cat':
            if self.pos == 'all':
                #match all sizees into layer3
                out_vis_1 = F.interpolate(out_vis_1, size=(out_vis.size(2), out_vis.size(3)), mode='bilinear')
                out_lwir_1 = F.interpolate(out_lwir_1, size=(out_lwir.size(2), out_lwir.size(3)), mode='bilinear')
                out_vis_2 = F.interpolate(out_vis_2, size=(out_vis.size(2), out_vis.size(3)), mode='bilinear')
                out_lwir_2 = F.interpolate(out_lwir_2, size=(out_lwir.size(2), out_lwir.size(3)), mode='bilinear')
                out = torch.cat((out_vis, out_lwir, out_vis_1, out_lwir_1, out_vis_2, out_lwir_2), 1)
                out = self.post_att(out)
                out = self.reduction_conv(out)
            elif self.pos == 'last':
                out = torch.cat((out_vis, out_lwir), 1)
                out = self.post_att(out)
                out = self.reduction_conv(out)
        
        elif self.fusion == 'add':
            out = out_vis + out_lwir
            out = self.post_att(out)
        
        else:
            out = torch.add(out_vis * 0.5, out_lwir * 0.5)

        # out_vis, out_lwir = self.insa(out_vis, out_lwir)
        # Weighted summation
        # out = torch.add(out_vis * self.weight, out_lwir * (1 - self.weight))
        # weight-sharing network
        out = self.pool3(out)

        out = F.relu(self.conv4_1_bn(self.conv4_1(out))) 
        out = F.relu(self.conv4_2_bn(self.conv4_2(out))) 
        out = F.relu(self.conv4_3_bn(self.conv4_3(out))) 
        out = self.pool4(out)
        conv4_3_feats = out
        
        norm = conv4_3_feats.pow(2).sum(dim=1, keepdim=True).sqrt()
        conv4_3_feats = conv4_3_feats / norm
        conv4_3_feats = conv4_3_feats * self.rescale_factors

        out = F.relu(self.conv5_1_bn(self.conv5_1(out))) 
        out = F.relu(self.conv5_2_bn(self.conv5_2(out))) 
        out = F.relu(self.conv5_3_bn(self.conv5_3(out))) 
        out = self.pool5(out)
        
        out = F.relu(self.conv6_1_bn(self.conv6_1(out)))
        out = F.relu(self.conv6_2(out))
        conv6_feats = out

        out = F.relu(self.conv7_1(out))
        out = F.relu(self.conv7_2_bn(self.conv7_2(out)))
        conv7_feats = out

        out = F.relu(self.conv8_1(conv7_feats))
        out = F.relu(self.conv8_2(out)) 
        conv8_feats = out  

        out = F.relu(self.conv9_1(out))
        out = F.relu(self.conv9_2(out))
        conv9_feats = out 

        out = F.relu(self.conv10_1(out)) 
        out = F.relu(self.conv10_2(out)) 
        conv10_feats = out
        
        return conv4_3_feats, conv6_feats, conv7_feats, conv8_feats, conv9_feats, conv10_feats


    def load_pretrained_layers(self):
        """
        As in the paper, we use a VGG-16 pretrained on the ImageNet task as the base network.
        There's one available in PyTorch, see https://pytorch.org/docs/stable/torchvision/models.html#torchvision.models.vgg16
        We copy these parameters into our network. It's straightforward for conv1 to conv5.
        However, the original VGG-16 does not contain the conv6 and con7 layers.
        Therefore, we convert fc6 and fc7 into convolutional layers, and subsample by decimation. See 'decimate' in utils.py.
        """

        # Current state of model
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG_BN
        pretrained_state_dict = torchvision.models.vgg16_bn(pretrained=True).state_dict()
        pretrained_param_names = list(pretrained_state_dict.keys())
        
        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names[1:50]):
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]

        for i, param in enumerate(param_names[50:99]):    
            if param == 'conv1_1_lwir.weight':
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]][:, :1, :, :]              
            else:
                state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]
        
        self.load_state_dict(state_dict)

        print("Load Model: ATTINSANet\n")



class ATTNet(INSANet):
    """
    The INSA network - encapsulates the INSA network.
    """

    def __init__(self, device, n_classes, attention =['CBAM', 'CHANNEL', 'SPATIAL'], pos = ['first', 'last', 'all'], fusion = ['add','cat']):
        super(ATTNet, self).__init__()
        # super().__init__(n_classes)
        self.device = device
        self.n_classes = n_classes
        self.base = VGGEdited(device, attention, pos, fusion)
        self.pred_convs = PredictionConvolutions(n_classes)
        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()

class ATTINSANet(INSANet):
    def __init__(self, device, n_classes, attention =['CBAM', 'CHANNEL', 'SPATIAL'], pos = ['first', 'last', 'all'], fusion = ['add','cat']):
        super(ATTINSANet, self).__init__(n_classes)
        self.device = device
        self.n_classes = n_classes
        self.base = VGGATTINSA(device, attention, pos, fusion)
        self.pred_convs = PredictionConvolutions(n_classes)
        # Prior boxes
        self.priors_cxcy = self.create_prior_boxes()
