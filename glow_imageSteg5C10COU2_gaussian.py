import torch
from torch import nn
from torch.nn import functional as F
from math import log, pi, exp
import numpy as np
logabs = lambda x: torch.log(torch.abs(x))
import config1 as c
import modules.module_util as mutil



class ActNorm(nn.Module):
    def __init__(self, in_channel):
        super().__init__()
        #只对channel纬度进行运算，也就是本质上loc,scale只有num_channels个数值
        self.loc = nn.Parameter(torch.zeros(1, in_channel, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, in_channel, 1, 1))
        self.initialized = nn.Parameter(torch.tensor(0,dtype=torch.uint8),requires_grad=False)

    def initialize(self, input):
        with torch.no_grad():
            #对除通道外的纬度进行计算均值和方差，保持原纬度不变，用于广播
            mean = torch.mean(input,dim=(0,2,3),keepdim=True)
            std = torch.std(input,dim=(0,2,3),keepdim=True)
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        batch_size, _, height, width = input.shape
        #item返回的是python格式的标量支队标量有效
        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)
        log_abs = logabs(self.scale)
        return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc
class InvConv2d(nn.Module):
    """
    Implementation of the invertible 1x1 convolution layer defined in 
    Glow: Generative Flow with Invertible 1x1 Convolutions
    (https://arxiv.org/abs/1807.03039).
    
    This code is a modified version of the repository
    https://github.com/rosinality/glow-pytorch
    
    """
    def __init__(self, num_channels, LU_decomposed=True, epsilon=1e-6):  
        super().__init__()  
        w_shape = [num_channels, num_channels]  
        w_init = np.linalg.qr(np.random.randn(*w_shape))[0].astype(np.float32)  
        if not LU_decomposed:  
            # Sample a random orthogonal matrix:  
            self.weight = nn.Parameter(torch.Tensor(w_init))  
        else:  
            np_p, np_l, np_u = splin.lu(w_init)  
            np_s = np.diag(np_u)  
            np_sign_s = np.sign(np_s)  
            np_log_s = np.log(np.abs(np_s))  
            np_u = np.triu(np_u, k=1)  
            l_mask = np.tril(np.ones(w_shape, dtype=np.float32), -1)  
            eye = np.eye(*w_shape, dtype=np.float32)  

            # 确保对角线元素非零  
            np_l += eye * epsilon  

            # 将这些变量注册为缓冲区  
            self.register_buffer('p', torch.Tensor(np_p.astype(np.float32)))  
            self.register_buffer('sign_s', torch.Tensor(np_sign_s.astype(np.float32)))  
            self.register_buffer('l_mask', torch.Tensor(l_mask))  
            self.register_buffer('eye', torch.Tensor(eye))  
            self.register_buffer('w_shape', torch.tensor(w_shape))  

            self.l = nn.Parameter(torch.Tensor(np_l.astype(np.float32)))  
            self.log_s = nn.Parameter(torch.Tensor(np_log_s.astype(np.float32)))  
            self.u = nn.Parameter(torch.Tensor(np_u.astype(np.float32)))  

            self.LU = LU_decomposed  
            self.epsilon = epsilon
    def get_weight(self, z, reverse):  
        w_shape = self.w_shape  
        pixels = z.shape[2] * z.shape[3]  
        if not self.LU:  
            dlogdet = torch.slogdet(self.weight)[1] * pixels  
            if not reverse:  
                weight = self.weight.view(w_shape[0], w_shape[1], 1, 1)  
            else:  
                weight = torch.inverse(self.weight.double()).float()\
                              .view(w_shape[0], w_shape[1], 1, 1)  
            return weight, dlogdet.repeat(z.shape[0])  
        else:  
            self.p = self.p.to(z.device)  
            self.sign_s = self.sign_s.to(z.device)  
            self.l_mask = self.l_mask.to(z.device)  
            self.eye = self.eye.to(z.device)  

            # 确保对角线元素非零  
            l = self.l * self.l_mask + self.eye + self.epsilon * self.eye  
            u = self.u * self.l_mask.transpose(0, 1).contiguous() + torch.diag(self.sign_s * torch.exp(self.log_s))  

            if not reverse:  
                w = torch.matmul(self.p, torch.matmul(l, u))  
            else:
                l = torch.inverse(l.double()).float()  
                u = torch.inverse(u.double()).float()  
                w = torch.matmul(u, torch.matmul(l, self.p.inverse()))
            return w.view(w_shape[0], w_shape[1], 1, 1) 
    def forward(self, z):
        weight = self.get_weight(z, False)
        z = F.conv2d(z, weight)
        return z

    def reverse(self, z):
        weight= self.get_weight(z, True)
        z = F.conv2d(z, weight)
        return z         
class ZeroConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, padding=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, padding=0)
        self.conv.weight.data.zero_()
        self.conv.bias.data.zero_()
        self.scale = nn.Parameter(torch.zeros(1, out_channel, 1, 1))

    def forward(self, input):
        out = F.pad(input, [1, 1, 1, 1], value=1)
        out = self.conv(out)
        out = out * torch.exp(self.scale * 3)

        return out

#仿射变换层
class ResidualDenseBlock_out(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock_out, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        # initialization
        mutil.initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5
class INV_block(nn.Module):
    def __init__(self, subnet_constructor=ResidualDenseBlock_out, clamp=c.clamp, harr=True, in_1=12, in_2=12):
        super().__init__()
        if harr:
            self.split_len1 = in_1 * 4
            self.split_len2 = in_2 * 4
        self.clamp = clamp
        # ρ
        self.r = subnet_constructor(self.split_len1, self.split_len2)
        # η
        self.y = subnet_constructor(self.split_len1, self.split_len2)
        # φ
        self.f = subnet_constructor(self.split_len2, self.split_len1)
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))
    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1),
                  x.narrow(1, self.split_len1, self.split_len2))

        if not rev:

            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = self.e(s1) * x2 + t1

        else:

            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / self.e(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)

        return torch.cat((y1, y2), 1)
class AffineCoupling(nn.Module):
    def __init__(self, in_channel):
        super().__init__()

        self.inv1 = INV_block()
    def forward(self, input):
        out = self.inv1(input)
        return out
    def reverse(self, output):
        out = self.inv1(output, rev=True)

        return out
class Flow(nn.Module):
    def __init__(self, in_channel, affine=True):
        super().__init__()

       # self.actnorm = ActNorm(in_channel)
        self.invconv1 = InvConv2d(in_channel)
        self.invconv2 = InvConv2d(in_channel)
        self.invconv3 = InvConv2d(in_channel)
        self.invconv4 = InvConv2d(in_channel)
        self.invconv5 = InvConv2d(in_channel)
        self.invconv6 = InvConv2d(in_channel)
        self.invconv7 = InvConv2d(in_channel)
        self.invconv8 = InvConv2d(in_channel)
        self.invconv9 = InvConv2d(in_channel)
        self.invconv10 = InvConv2d(in_channel)

        
        self.coupling1 = AffineCoupling(in_channel)
        self.coupling2 = AffineCoupling(in_channel)
    def forward(self, input):
        #out= self.actnorm(input)
        #print(f"after actnorm :{out.cpu().shape}")
        out= self.invconv1(input)
        out= self.invconv2(out)
        
        out= self.invconv3(out)
        out= self.invconv4(out)
        out= self.invconv5(out)
        out= self.invconv6(out)
        out= self.invconv7(out)
        out= self.invconv8(out)
        out= self.invconv9(out)     
                     
        out= self.invconv10(out)            
        out, _ = self.attn(out)
        out= self.coupling1(out)           
        #print(f"after invconv :{out.cpu().shape}")
        out= self.coupling2(out)           
        #print(f"after coupling :{out.cpu().shape}")
        return out

    def reverse(self, output):
        #print("before coupling",output.cpu())
        input = self.coupling2.reverse(output)
        input= self.coupling1.reverse(input)
     
        
        input = self.invconv10.reverse(input)
        input = self.invconv9.reverse(input)
        input = self.invconv8.reverse(input)
        input = self.invconv7.reverse(input)
        input = self.invconv6.reverse(input)
        input = self.invconv5.reverse(input)
        input = self.invconv4.reverse(input)
        input = self.invconv3.reverse(input)
        input = self.invconv2.reverse(input)
        input = self.invconv1.reverse(input)
        #print("before invconv",input.cpu())
        #input = self.invconv.reverse(input)
        #print("before actnorm",input.cpu())
        #input = self.actnorm.reverse(input)

        return input
class Block(nn.Module):
    def __init__(self, in_channel, n_flow, split=True, affine=True, conv_lu=True):
        super().__init__()

        squeeze_dim = in_channel * 4

        self.flows = nn.ModuleList()
        for i in range(n_flow):
            self.flows.append(Flow(squeeze_dim))

        self.split = split
    def forward(self, input):
        out = input

        for flow in self.flows:
            out= flow(out)

        return out

    def reverse(self, output):
        input = output
        for flow in self.flows[::-1]:
            input = flow.reverse(input)
        return input

class Glow(nn.Module):
    def __init__(
        self, in_channel, n_flow, n_block, affine=True
    ):
        super().__init__()

        self.blocks = nn.ModuleList()
        n_channel = in_channel
        for i in range(n_block):
            self.blocks.append(Block(n_channel, n_flow, affine=affine))
    def forward(self, input):
        log_p_sum = 0
        out = input
        z_outs = []
        b_size, n_channel, height, width = input.shape
        squeezed = input.view(b_size, n_channel, height // 2, 2, width // 2, 2)
        squeezed = squeezed.permute(0, 1, 3, 5, 2, 4)
        out = squeezed.contiguous().view(b_size, n_channel * 4, height // 2, width // 2)
        #print(f"out.shape is {out.shape}")
        for block in self.blocks:
            out = block(out)
        out,r_loss = out.chunk(2, 1)
        return out,r_loss

    def reverse(self, output,z):
        input = output
        input = torch.cat([output, z], 1)
        for i, block in enumerate(self.blocks[::-1]):
                input = block.reverse(input)
                b_size, n_channel, height, width = input.shape
        unsqueezed = input.view(b_size, n_channel // 4, 2, 2, height, width)
        unsqueezed = unsqueezed.permute(0, 1, 4, 2, 5, 3)
        unsqueezed = unsqueezed.contiguous().view(
            b_size, n_channel // 4, height * 2, width * 2
        )
        return unsqueezed