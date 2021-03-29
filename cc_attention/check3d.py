import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax
import torch
import torch.nn as nn

import torch.autograd as autograd
import torch.cuda.comm as comm
import torch.nn.functional as F
from torch.autograd.function import once_differentiable
from torch.utils.cpp_extension import load
import os, time
import functools
from torch.autograd import Variable


def INF3DH(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(H),0).unsqueeze(0).repeat(B*W*D,1,1).cuda()
def INF3DW(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(W),0).unsqueeze(0).repeat(B*H*D,1,1).cuda()
def INF3DD(B, H, W, D):
    return -torch.diag(torch.tensor(float("inf")).repeat(D),0).unsqueeze(0).repeat(B*H*W,1,1).cuda()
class CrissCrossAttention3D(nn.Module):
    """ Criss-Cross Attention Module 3D version, inspired by the 2d version"""
    def __init__(self,  verbose = False):
        super(CrissCrossAttention3D,self).__init__()
        self.softmax = Softmax(dim=4)
        self.INFH = INF3DH
        self.INFD = INF3DD
        self.verbose = verbose


    def forward(self, proj_query,proj_key,proj_value):
        m_batchsize, _, height, width, depth= proj_query.size()
        #proj_query = self.query_conv(x)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize*width*depth,-1,height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize*height*depth,-1,width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize*height*width,-1,depth).permute(0, 2, 1)
        
        if self.verbose: print_tensor('q', proj_query)
        if self.verbose: print_tensor('qh', proj_query_H)
        if self.verbose: print_tensor('qw', proj_query_W)
        if self.verbose: print_tensor('qd', proj_query_D)

        #proj_key = self.key_conv(x)

        # bchw > bwch, b*w*d-c-h
        proj_key_H = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
        # bchw > bhcw, b*h*d-c-w
        proj_key_W = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
        proj_key_D = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#b*h*w-c-d

        if self.verbose: print_tensor('k', proj_key)
        if self.verbose: print_tensor('kh', proj_key_H)
        if self.verbose: print_tensor('kw', proj_key_W)
        if self.verbose: print_tensor('kd', proj_key_D)

        #proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)#bchwd->bwdch
        proj_value_W = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)#bchwd->bhdcw
        proj_value_D = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#bchwd->bhwcd

        # batch matrix-matrix
        inf_holder = self.INFH(m_batchsize, height, width, depth) # > bw-h-h 
        if self.verbose: print_tensor('inf', inf_holder)
        energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize,width,depth,height,height).permute(0,1,3,2,4) # bwhdh
        if self.verbose: print_tensor('eh', energy_H) 

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(proj_query_W, proj_key_W)#+self.INFW(m_batchsize, height, width, depth)
        energy_W = energy_W.view(m_batchsize, height, depth, width, width).permute(0, 3, 1, 2, 4) # bwhdw
        if self.verbose: print_tensor('ew', energy_W)
        
        #  b*h*w-d-c, b*h*w-c-d > b*h*w-d-d
        energy_D = (torch.bmm(proj_query_D, proj_key_D)+self.INFD(m_batchsize, height, width, depth)).view(m_batchsize, height, width, depth, depth).permute(0, 2, 1, 3, 4)# bwhdd
        if self.verbose: print_tensor('ew', energy_W)


        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4)) # bwhd*(h+w+d)
        if self.verbose: print_tensor('eall', concate) 
        # bhw(H+W) > bhwH, bwhH; 
        att_H = concate[:,:,:,:,0:height].permute(0,1,3,2,4).contiguous().view(m_batchsize*width*depth,height,height)
        att_W = concate[:,:,:,:,height:height+width].permute(0,2,3,1,4).contiguous().view(m_batchsize*height*depth,width,width)
        att_D = concate[:,:,:,:,height+width:].permute(0,2,1,3,4).contiguous().view(m_batchsize*height*width, depth, depth)

        if self.verbose: print_tensor('atth', att_H); print_tensor('attw', att_W);print_tensor('attd', att_D)

        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,depth,-1,height).permute(0,3,4,1,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,depth,-1, width).permute(0,3,1,4,2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize,height, width, -1, depth).permute(0,3,1,2,4)

        if self.verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
        #print(out_H.size(),out_W.size())
        return out_H + out_W + out_D

class CrissCrossAttention3D1(nn.Module):
    """ Criss-Cross Attention Module 3D version, inspired by the 2d version"""
    def __init__(self,  verbose = False):
        super(CrissCrossAttention3D1,self).__init__()
        self.softmax = Softmax(dim=4)
        self.INFH = INF3DH
        self.INFW = INF3DW
        self.verbose = verbose


    def forward(self, proj_query,proj_key,proj_value):
        m_batchsize, _, height, width, depth= proj_query.size()
        #proj_query = self.query_conv(x)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize*width*depth,-1,height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize*height*depth,-1,width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize*height*width,-1,depth).permute(0, 2, 1)
        
        if self.verbose: print_tensor('q', proj_query)
        if self.verbose: print_tensor('qh', proj_query_H)
        if self.verbose: print_tensor('qw', proj_query_W)
        if self.verbose: print_tensor('qd', proj_query_D)

        #proj_key = self.key_conv(x)

        # bchw > bwch, b*w*d-c-h
        proj_key_H = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
        # bchw > bhcw, b*h*d-c-w
        proj_key_W = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
        proj_key_D = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#b*h*w-c-d

        if self.verbose: print_tensor('k', proj_key)
        if self.verbose: print_tensor('kh', proj_key_H)
        if self.verbose: print_tensor('kw', proj_key_W)
        if self.verbose: print_tensor('kd', proj_key_D)

        #proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)#bchwd->bwdch
        proj_value_W = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)#bchwd->bhdcw
        proj_value_D = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#bchwd->bhwcd

        # batch matrix-matrix
        inf_holder = self.INFH(m_batchsize, height, width, depth) # > bw-h-h 
        if self.verbose: print_tensor('inf', inf_holder)
        energy_H = torch.bmm(proj_query_H, proj_key_H)+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize,width,depth,height,height).permute(0,3,1,2,4) # bhwdh
        if self.verbose: print_tensor('eh', energy_H) 

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(proj_query_W, proj_key_W)+self.INFW(m_batchsize, height, width, depth)
        energy_W = energy_W.view(m_batchsize, height, depth, width, width).permute(0, 1, 3, 2, 4) # bhwdw
        if self.verbose: print_tensor('ew', energy_W)
        # b*h*w-d-c ,b*h*w-c-d ->b*h*w-d-d
        energy_D = (torch.bmm(proj_query_D, proj_key_D)).view(m_batchsize, height, width, depth, depth)#.permute(0, 1, 3, 2, 4)#+self.INFD(m_batchsize, height, width, depth)
        if self.verbose: print_tensor('ew', energy_W)


        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4)) # bhwd*(h+w+d)
        if self.verbose: print_tensor('eall', concate) 
        # bhw(H+W) > bhwH, bwhH; 
        att_H = concate[:,:,:,:,0:height].permute(0,2,3,1,4).contiguous().view(m_batchsize*width*depth,height,height)
        att_W = concate[:,:,:,:,height:height+width].permute(0,1,3,2,4).contiguous().view(m_batchsize*height*depth,width,width)
        att_D = concate[:,:,:,:,height+width:].contiguous().view(m_batchsize*height*width, depth, depth)

        if self.verbose: print_tensor('atth', att_H); print_tensor('attw', att_W);print_tensor('attd', att_D)

        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,depth,-1,height).permute(0,3,4,1,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,depth,-1, width).permute(0,3,1,4,2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize,height, width, -1, depth).permute(0,3,1,2,4)

        if self.verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
        #print(out_H.size(),out_W.size())
        return out_H + out_W + out_D

class CrissCrossAttention3D2(nn.Module):
    """ Criss-Cross Attention Module 3D version, inspired by the 2d version"""
    def __init__(self,  verbose = False):
        super(CrissCrossAttention3D2,self).__init__()
        self.softmax = Softmax(dim=4)
        self.INFD = INF3DD
        self.INFW = INF3DW
        self.verbose = verbose


    def forward(self, proj_query,proj_key,proj_value):
        m_batchsize, _, height, width, depth= proj_query.size()
        #proj_query = self.query_conv(x)
        # bchw > bwch, b*w*d-c-h > b*w*d-h-c
        proj_query_H = proj_query.permute(0, 3, 4, 1, 2).contiguous().view(m_batchsize*width*depth,-1,height).permute(0, 2, 1)
        # bchw > bhcw, b*h*d-c-w > b*h*d-w-c
        proj_query_W = proj_query.permute(0, 2, 4, 1, 3).contiguous().view(m_batchsize*height*depth,-1,width).permute(0, 2, 1)
        # bchwd > bwch, b*h*w-c-d > b*h*w-d-c
        proj_query_D = proj_query.permute(0, 2, 3, 1, 4).contiguous().view(m_batchsize*height*width,-1,depth).permute(0, 2, 1)
        
        if self.verbose: print_tensor('q', proj_query)
        if self.verbose: print_tensor('qh', proj_query_H)
        if self.verbose: print_tensor('qw', proj_query_W)
        if self.verbose: print_tensor('qd', proj_query_D)

        #proj_key = self.key_conv(x)

        # bchw > bwch, b*w*d-c-h
        proj_key_H = proj_key.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)
        # bchw > bhcw, b*h*d-c-w
        proj_key_W = proj_key.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)
        proj_key_D = proj_key.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#b*h*w-c-d

        if self.verbose: print_tensor('k', proj_key)
        if self.verbose: print_tensor('kh', proj_key_H)
        if self.verbose: print_tensor('kw', proj_key_W)
        if self.verbose: print_tensor('kd', proj_key_D)

        #proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,4,1,2).contiguous().view(m_batchsize*width*depth,-1,height)#bchwd->bwdch
        proj_value_W = proj_value.permute(0,2,4,1,3).contiguous().view(m_batchsize*height*depth,-1,width)#bchwd->bhdcw
        proj_value_D = proj_value.permute(0,2,3,1,4).contiguous().view(m_batchsize*height*width,-1,depth)#bchwd->bhwcd

        # batch matrix-matrix
        #inf_holder = self.INFH(m_batchsize, height, width, depth) # > bw-h-h 
        #if self.verbose: print_tensor('inf', inf_holder)
        energy_H = torch.bmm(proj_query_H, proj_key_H)#+inf_holder # bwd-h-c, bwd-c-h > bwd-h-h
        energy_H = energy_H.view(m_batchsize,width,depth,height,height).permute(0,3,1,2,4) # bhwdh
        if self.verbose: print_tensor('eh', energy_H) 

        #  b*h*d-w-c, b*h*d-c-w > b*h*d-w-w
        energy_W = torch.bmm(proj_query_W, proj_key_W)+self.INFW(m_batchsize, height, width, depth)
        energy_W = energy_W.view(m_batchsize, height, depth, width, width).permute(0, 1, 3, 2, 4) # bhwdw
        if self.verbose: print_tensor('ew', energy_W)
        # b*h*w-d-c ,b*h*w-c-d ->b*h*w-d-d
        energy_D = (torch.bmm(proj_query_D, proj_key_D)+self.INFD(m_batchsize, height, width, depth)).view(m_batchsize, height, width, depth, depth)#.permute(0, 1, 3, 2, 4)#+self.INFD(m_batchsize, height, width, depth)
        if self.verbose: print_tensor('ew', energy_W)


        concate = self.softmax(torch.cat([energy_H, energy_W, energy_D], 4)) # bhwd*(h+w+d)
        if self.verbose: print_tensor('eall', concate) 
        # bhw(H+W) > bhwH, bwhH; 
        att_H = concate[:,:,:,:,0:height].permute(0,2,3,1,4).contiguous().view(m_batchsize*width*depth,height,height)
        att_W = concate[:,:,:,:,height:height+width].permute(0,1,3,2,4).contiguous().view(m_batchsize*height*depth,width,width)
        att_D = concate[:,:,:,:,height+width:].contiguous().view(m_batchsize*height*width, depth, depth)

        if self.verbose: print_tensor('atth', att_H); print_tensor('attw', att_W);print_tensor('attd', att_D)

        # p-c-h, p-h-h > p-c-h
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,depth,-1,height).permute(0,3,4,1,2)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,depth,-1, width).permute(0,3,1,4,2)
        out_D = torch.bmm(proj_value_D, att_D.permute(0, 2, 1)).view(m_batchsize,height, width, -1, depth).permute(0,3,1,2,4)

        if self.verbose: print_tensor('outh', out_H); print_tensor('outw', out_W), print_tensor('outd', out_D)
        #print(out_H.size(),out_W.size())
        return out_H + out_W + out_D
if __name__ == "__main__":

    ca1 = CrissCrossAttention3D2().cuda()
    x1 = Variable(torch.zeros(1, 8, 10, 10,1).cuda() + 1, requires_grad=True)
    y1 =Variable(torch.zeros(1, 8, 10, 10,1).cuda() + 2, requires_grad=True)
    z1 = Variable(torch.zeros(1, 64, 10, 10,1).cuda() + 3, requires_grad=True)
    out1 = ca1(x1, y1, z1)
    out1.sum().backward()
    print(x1.grad,y1.grad,z1.grad)
    print('\n')
    print("%.10f" %z1.grad[0,0,0,0,0])
    print (out1)
