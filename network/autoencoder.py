import torch
import torch.nn as nn
from collections import OrderedDict
from torch.nn import functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
 
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
   
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 256:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        # print("conv1",out.shape)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # print("conv2",out.shape)
        # print("downsample",self.downsample)
        if self.downsample is not None:
            identity = self.downsample(x)
            # print("downsample",identity.shape)

        out += identity
        out = self.relu(out)
        # print("out of basic block",out.shape)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class AutoEncoder(nn.Module):
        def __init__(self,groups=1,width_per_group=256):
            super(AutoEncoder, self).__init__()
            self._norm_layer =  nn.BatchNorm2d
            self.dilation = 1
            self.inplanes = 256
            self.groups = groups
            self.base_width = width_per_group

           
            i=0
            layers_enc=[]
            layers_dec=[]
            factor=self.inplanes
            for i in range(3):
                 factor/=4
                 block = self._make_layer(BasicBlock,int(factor)  ,1,stride = 1, dilate=True)
                #  print(block)
                 layers_enc.append(block)
                #  print('block : ',i)
                 
            i=0
            
            for i in range(3):     
                factor*=4
                layers_dec.append(self._make_layer(BasicBlock, int(factor) ,1,stride = 1, dilate=True))
            self.encoder= nn.Sequential(*layers_enc)
            self.decoder =nn.Sequential(*layers_dec)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print(m)
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        
                   

        def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                norm_layer = self._norm_layer
                downsample = None
                previous_dilation = self.dilation
                if block == BasicBlock:
                    previous_dilation = 1
                    self.dilation = 1 
                if dilate:
                    self.dilation *= stride
                    stride = 1
        
                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        norm_layer(planes * block.expansion),
                    )

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))

                return nn.Sequential(*layers)
        def forward(self, feat_x,feat_coco,mode =0):
            x_enc=self.encoder(feat_x)
             
            if mode == 0 :
                    with torch.no_grad():
                        x_coco = self.encoder(feat_coco) 
                        u,_,v = torch.linalg.svd(x_enc)
            
                        s2= torch.linalg.svdvals(x_coco) 

                        x_enc_rand= u @ torch.diag_embed(s2) @ v
            else:
                    x_enc_rand=x_enc
            #  print(x_enc_rand.shape)
            #  print(x_enc_rand)
            x_dec =self.decoder(x_enc_rand)
            #  print(x_dec.shape)
            return x_dec, x_enc,x_enc_rand
        
          
class AutoencoderBottleNet(nn.Module):
  def __init__(self,groups=1,width_per_group=256):
            super(AutoencoderBottleNet, self).__init__()
            self._norm_layer =  nn.BatchNorm2d
            self.dilation = 1
            self.inplanes = 256
            self.groups = groups
            self.base_width = width_per_group


            i=0
            layers_enc=[]
            layers_dec=[]
            factor=self.inplanes
            for i in range(4):
                 factor/=4
                 block = self._make_layer(Bottleneck,int(factor)  ,1,stride = 1, dilate=True)
                 print(block)
                 print(factor)
                 layers_enc.append(block)
                #  print('block : ',i)

            i=0

            for i in range(3):
                factor*=4
                print(block)
                print(factor)
                layers_dec.append(self._make_layer(Bottleneck, int(factor) ,1,stride = 1, dilate=True))
            self.encoder= nn.Sequential(*layers_enc)
            self.decoder =nn.Sequential(*layers_dec)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print(m)
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)



  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                norm_layer = self._norm_layer
                downsample = None
                previous_dilation = self.dilation
                if block == BasicBlock:
                    previous_dilation = 1
                    self.dilation = 1
                if dilate:
                    self.dilation *= stride
                    stride = 1

                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        norm_layer(planes * block.expansion),
                    )
                    print("downsample",downsample, self.inplanes, planes * block.expansion, stride)

                layers = []
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))

                return nn.Sequential(*layers)
  def forward(self, feat_x,feat_coco,mode =0):
            x_enc=self.encoder(feat_x)
            # print('encoder shape',x_enc.shape)

            if mode == 0 :
                    with torch.no_grad():
                        x_coco = self.encoder(feat_coco)
                        u,_,v = torch.linalg.svd(x_enc)

                        s2= torch.linalg.svdvals(x_coco)

                        x_enc_rand= u @ torch.diag_embed(s2) @ v
            else:
                    x_enc_rand=x_enc
            #  print(x_enc_rand.shape)
            #  print(x_enc_rand)
            x_dec =self.decoder(x_enc_rand)

            # print("dec shape",x_dec.shape)
            return x_dec, x_enc,x_enc_rand




class AutoencoderHighResolution(nn.Module):
  def __init__(self,groups=1,width_per_group=256):
            super(AutoencoderHighResolution, self).__init__()
            self._norm_layer =  nn.BatchNorm2d
            self.dilation = 1
            self.inplanes = 256
            self.groups = groups
            self.base_width = width_per_group


            i=0
            layers_enc=[]
            layers_dec=[]
            factor=self.inplanes
            for i in range(4):
                 factor/=4
                 block = self._make_layer(Bottleneck,int(factor)  ,1,stride = 1, dilate=True)
                 #print(block)
                 #print(factor)
                 layers_enc.append(block)
                #  print('block : ',i)

            i=0

            for i in range(3):
                factor*=4
                if i == 2:

                  block=self._make_layer(block=Bottleneck,planes= int(factor) ,blocks=1,stride = 1, dilate=True)
                else:
                  block=self._make_layer(block=Bottleneck,planes= int(factor) ,blocks=1,stride = 2, dilate=False)

                print(block)
                print(factor)
                layers_dec.append(block)
            self.encoder= nn.Sequential(*layers_enc)
            self.decoder =nn.Sequential(*layers_dec)
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    # print(m)
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)



  def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
                norm_layer = self._norm_layer
                downsample = None
                print("stride", stride)
                previous_dilation = self.dilation
                if block == BasicBlock:
                    previous_dilation = 1
                    self.dilation = 1
                if dilate:
                    self.dilation *= stride
                    stride = 1

                if stride != 1 or self.inplanes != planes * block.expansion:
                    downsample = nn.Sequential(
                        conv1x1(self.inplanes, planes * block.expansion, stride),
                        norm_layer(planes * block.expansion),
                    )
                    #print("downsample",downsample, self.inplanes, planes * block.expansion, stride)

                layers = []
                # print("stride", stride)
                layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                                    self.base_width, previous_dilation, norm_layer))
                self.inplanes = planes * block.expansion
                for _ in range(1, blocks):
                    layers.append(block(self.inplanes, planes, groups=self.groups,
                                        base_width=self.base_width, dilation=self.dilation,
                                        norm_layer=norm_layer))

                return nn.Sequential(*layers)
  def forward(self, feat_x,feat_coco,mode =0):
            x_enc=feat_x
            input_shape = x_enc.shape[2:]
            #print(input_shape)
            #print((input_shape[0]*2,input_shape[1]*2))
            factor =1
            for i in range(len(self.encoder)):
              x_enc = self.encoder[i](x_enc)
              if(input_shape[0]*factor<= 768):
                x_enc =F.interpolate(x_enc,size=(input_shape[0]*factor,input_shape[1]*factor), mode='bilinear',align_corners=False)
                factor*=2
              print('encoder '+str(i)+'shape',x_enc.shape)


            #x_enc=self.encoder(feat_x)
            print('encoder shape',x_enc.shape)

            if mode == 0 :
                    with torch.no_grad():
                        x_coco = feat_coco
                        factor =1
                        for i in range(len(self.encoder)):
                          x_coco = self.encoder[i](x_coco)
                          if(input_shape[0]*factor<= 768):
                            x_coco =F.interpolate(x_coco,size=(input_shape[0]*factor,input_shape[1]*factor), mode='bilinear',align_corners=False)
                            factor*=2
                          print('x_coco shape',x_coco.shape)
                        u,_,v = torch.linalg.svd(x_enc)

                        s2= torch.linalg.svdvals(x_coco)

                        x_enc_rand= u @ torch.diag_embed(s2) @ v
            else:
                    x_enc_rand=x_enc
            #  print(x_enc_rand.shape)
            #  print(x_enc_rand)

            x_dec =self.decoder(x_enc_rand)

            print("dec shape",x_dec.shape)
            return x_dec, x_enc,x_enc_rand



m=AutoencoderHighResolution()
# m=AutoencoderBottleNet()
print(m)
# m=AutoEncoder()
x=torch.rand(256,192,192).unsqueeze(0)
x1=torch.rand(256,192,192).unsqueeze(0)
print(m(x,x)[0].shape)



# out=m(x,x1)