import torch
import torch.nn as nn

class Conv2ResidualSame(nn.Module) :
    # input channel and output channel are smae
    def __init__(self, ch_size , k_size) :
        super(Conv2ResidualSame , self).__init__()
        pad_size = int(k_size/2)
        self.c_res = nn.Sequential(nn.Conv2d(ch_size,ch_size,k_size,padding=pad_size,stride=1),
                                   nn.ReLU(),
                                   nn.Conv2d(ch_size,ch_size,k_size,padding=pad_size,stride=1))
        self.init_param()

    def init_param(self) :
        for m in self.modules() :
            if isinstance(m , nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self , in_tensor) :
        out_tensor = self.c_res(in_tensor)
        res_tensor = torch.add(out_tensor, in_tensor)
        return res_tensor

class Conv2ResidualDiff(nn.Module) :
    # input channel and output channel are different
    def __init__(self, in_ch , out_ch , k_size) :
        super(Conv2ResidualDiff , self).__init__()
        pad_size = int(k_size/2)
        # stride size is 2 and channel size is doubled 
        self.c_res = nn.Sequential(nn.Conv2d(in_ch,out_ch,k_size,padding=pad_size,stride=2),
                                   nn.ReLU(),
                                   nn.Conv2d(out_ch,out_ch,k_size,padding=pad_size,stride=1))
        # 1 X 1 Convolution to match input dimension and out dimension
        self.conv11 = nn.Conv2d(in_ch,out_ch,1,stride=2) 

        self.init_param()

    def init_param(self) :
        for m in self.modules() :
            if isinstance(m , nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self , in_tensor) :
        out_tensor = self.c_res(in_tensor)
        project_tensor = self.conv11(in_tensor)
        res_tensor = torch.add(out_tensor, project_tensor)
        return res_tensor
    
class Conv2Block(nn.Module) :
    def __init__(self, layer_size , in_ch , ch_size , k_size) :
        super(Conv2Block , self).__init__()
        self.layer_size = layer_size
        self.c_block = nn.ModuleList()
        # Convolutional Residual block (output channel is doubled and feature map size is halvec) 
        self.c_block.append(Conv2ResidualDiff(in_ch,ch_size,k_size))
        # Convolutional Residual block (input channel and output channel is same and also with feature map size)
        for i in range(layer_size) :
            self.c_block.append(Conv2ResidualSame(ch_size,k_size))

    def forward(self , in_tensor) :
        tensor_ptr = in_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.c_block[i](tensor_ptr)
        return tensor_ptr

class ResNet(nn.Module) :
    def __init__(self, layer_list , image_size , ch_list , 
                 in_kernal , kernal_size , class_size) :
        super(ResNet , self).__init__()

        self.ch_list = ch_list
        self.resnet = nn.ModuleList()
        size_ptr = image_size
        # first convolutional layer
        self.resnet.append(nn.Conv2d(3,ch_list[0],in_kernal,stride=2,padding=int(in_kernal/2))) 
        size_ptr /= 2

        ch_ptr = ch_list[0]
        for i in range(len(layer_list)) :
            # convolution residual block
            convblock = Conv2Block(layer_list[i], ch_ptr, ch_list[i], kernal_size)
            self.resnet.append(convblock) 
            self.resnet.append(nn.BatchNorm2d(ch_list[i])) # batch normalization

            size_ptr /= 2
            ch_ptr = ch_list[i]

        self.avg_pool = nn.AvgPool2d(int(size_ptr)) # final average pooling layer
        self.o_layer = nn.Linear(ch_list[-1], class_size)
        self.layer_size = len(self.resnet)

        self.init_param()

    def init_param(self) :
        nn.init.kaiming_normal_(self.o_layer.weight)
        nn.init.zeros_(self.o_layer.bias) 

    def forward(self , in_tensor) :
        batch_size = in_tensor.shape[0]
        tensor_ptr = in_tensor
        for i in range(self.layer_size) :
            tensor_ptr = self.resnet[i](tensor_ptr)

        avg_tensor = self.avg_pool(tensor_ptr)
        avg_tensor = torch.reshape(avg_tensor , (batch_size , self.ch_list[-1]))
        o_tensor = self.o_layer(avg_tensor)
        return o_tensor
