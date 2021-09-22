import torch
import torch.nn as nn

class ConvBlock(nn.Module) :
    def __init__(self, ch_size, k_size) :
        super(ConvBlock, self).__init__() 
        pad_size = int(k_size/2)
        self.conv_net = nn.Sequential(
            nn.Conv2d(ch_size, ch_size, k_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(ch_size),
            nn.ReLU(),
            nn.Conv2d(ch_size, ch_size, k_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(ch_size)    
        )
        self.relu = nn.ReLU()
        
    def forward(self, in_tensor) :
        h_tensor = self.conv_net(in_tensor)
        o_tensor = self.relu(h_tensor + in_tensor)
        return o_tensor

class Transition(nn.Module) :
    def __init__(self, in_ch, out_ch, k_size) :
        super(Transition, self).__init__()
        pad_size = int(k_size/2)
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k_size, padding=pad_size, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, k_size, padding=pad_size, stride=1),
            nn.BatchNorm2d(out_ch)
        )
        self.proj = nn.Conv2d(in_ch, out_ch, 1, stride=2) 
        self.relu = nn.ReLU()

    def forward(self, in_tensor) :
        h_tensor = self.conv_net(in_tensor)
        x_tensor = self.proj(in_tensor)
        o_tensor = self.relu(h_tensor + x_tensor)
        return o_tensor

class ResNetBlock(nn.Module) :
    def __init__(self, layer_size, in_ch, out_ch, k_size) :
        super(ResNetBlock, self).__init__()
        self.layer_size = layer_size
        self.res_block = nn.ModuleList()
        self.res_block.append(Transition(in_ch, out_ch, k_size))
        for i in range(layer_size-1) :
            self.res_block.append(ConvBlock(out_ch, k_size))
    
    def forward(self, in_tensor) :
        tensor_ptr = in_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.res_block[i](tensor_ptr)
        return tensor_ptr

class ResNet(nn.Module) :
    def __init__(self, 
        layer_list ,  
        ch_list , 
        image_size ,
        in_channel ,
        in_kernal , 
        kernal_size , 
        class_size) :
        super(ResNet , self).__init__()

        self.ch_list = ch_list
        self.resnet = nn.ModuleList()
        self.resnet.append(nn.Conv2d(3, in_channel, in_kernal, stride=2, padding=int(in_kernal/2)))
        self.resnet.append(nn.MaxPool2d(3, stride=2, padding=1)) 
        size_ptr = image_size/4

        for i in range(layer_list[0]) :
            self.resnet.append(ConvBlock(ch_list[0], kernal_size))
        
        for i in range(1, len(layer_list)) :
            self.resnet.append(ResNetBlock(layer_list[i], ch_list[i-1], ch_list[i], kernal_size))
            size_ptr /= 2
            
        self.avg_pool = nn.AvgPool2d(int(size_ptr)) # final average pooling layer
        self.o_layer = nn.Linear(ch_list[-1], class_size)
        self.layer_size = len(self.resnet)

        self.init_param()

    def init_param(self) :
        for p in self.parameters() :
            if p.dim() > 1 :
                nn.init.kaiming_uniform_(p)
        
    def forward(self , in_tensor) :
        batch_size = in_tensor.shape[0]
        tensor_ptr = in_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.resnet[i](tensor_ptr)
            
        avg_tensor = self.avg_pool(tensor_ptr)
        avg_tensor = torch.reshape(avg_tensor , (batch_size , self.ch_list[-1]))

        o_tensor = self.o_layer(avg_tensor)
        return o_tensor
