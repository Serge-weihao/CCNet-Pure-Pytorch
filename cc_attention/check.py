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
curr_dir = os.path.dirname(os.path.abspath(__file__))
_src_path = os.path.join(curr_dir, "src")
_build_path = os.path.join(curr_dir, "build")
os.makedirs(_build_path, exist_ok=True)
rcca = load(name="rcca",
            extra_cflags=["-O3"],
            build_directory=_build_path,
            verbose=True,
            sources = [os.path.join(_src_path, f) for f in [
                "lib_cffi.cpp", "ca.cu"
                ]],
            extra_cuda_cflags=["--expt-extended-lambda"])

def _check_contiguous(*args):
    if not all([mod is None or mod.is_contiguous() for mod in args]):
        raise ValueError("Non-contiguous input")


class CA_Weight(autograd.Function):
    @staticmethod
    def forward(ctx, t, f):
        # Save context
        n, c, h, w = t.size()
        size = (n, h+w-1, h, w)
        weight = torch.zeros(size, dtype=t.dtype, layout=t.layout, device=t.device)

        rcca.ca_forward_cuda(t, f, weight)
        
        # Output
        ctx.save_for_backward(t, f)

        return weight

    @staticmethod
    @once_differentiable
    def backward(ctx, dw):
        t, f = ctx.saved_tensors

        dt = torch.zeros_like(t)
        df = torch.zeros_like(f)

        rcca.ca_backward_cuda(dw.contiguous(), t, f, dt, df)

        _check_contiguous(dt, df)

        return dt, df

class CA_Map(autograd.Function):
    @staticmethod
    def forward(ctx, weight, g):
        # Save context
        out = torch.zeros_like(g)
        rcca.ca_map_forward_cuda(weight, g, out)
        
        # Output
        ctx.save_for_backward(weight, g)

        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, dout):
        weight, g = ctx.saved_tensors

        dw = torch.zeros_like(weight)
        dg = torch.zeros_like(g)

        rcca.ca_map_backward_cuda(dout.contiguous(), weight, g, dw, dg)

        _check_contiguous(dw, dg)

        return dw, dg

ca_weight = CA_Weight.apply
ca_map = CA_Map.apply
class CrissCross(nn.Module):
    """ Criss-Cross Attention Module"""
    def __init__(self):
        super(CrissCross,self).__init__()

    def forward(self, proj_query,proj_key,proj_value):
        energy = ca_weight(proj_query, proj_key)
        attention = F.softmax(energy, 1)
        out = ca_map(attention, proj_value)
        return out
def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)
class CC(nn.Module):
    def __init__(self):
        super(CC, self).__init__()
        self.softmax = Softmax(dim=3)
        self.INF = INF

    def forward(self, proj_query,proj_key,proj_value):
        m_batchsize, _, height, width = proj_query.size()
        #proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height).permute(0, 2, 1)
        proj_query_W = proj_query.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width).permute(0, 2, 1)
        #proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_key_W = proj_key.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        #proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0,3,1,2).contiguous().view(m_batchsize*width,-1,height)
        proj_value_W = proj_value.permute(0,2,1,3).contiguous().view(m_batchsize*height,-1,width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H)+self.INF(m_batchsize, height, width)).view(m_batchsize,width,height,height).permute(0,2,1,3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize,height,width,width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))
        #concate = concate * (concate>torch.mean(concate,dim=3,keepdim=True)).float()
        att_H = concate[:,:,:,0:height].permute(0,2,1,3).contiguous().view(m_batchsize*width,height,height)
        #print(concate)
        #print(att_H) 
        att_W = concate[:,:,:,height:height+width].contiguous().view(m_batchsize*height,width,width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize,width,-1,height).permute(0,2,3,1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize,height,-1,width).permute(0,2,1,3)
        #print(out_H.size(),out_W.size())
        return out_H + out_W
if __name__ == "__main__":

    ca = CrissCross().cuda()
    ca1 = CC().cuda()
    x = Variable(torch.zeros(1, 8, 10, 10).cuda() + 1, requires_grad=True)
    y = Variable(torch.zeros(1, 8, 10, 10).cuda() + 2, requires_grad=True)
    z = Variable(torch.zeros(1, 64, 10, 10).cuda() + 3, requires_grad=True)
    x1 = Variable(torch.zeros(1, 8, 10, 10).cuda() + 1, requires_grad=True)
    y1 =Variable(torch.zeros(1, 8, 10, 10).cuda() + 2, requires_grad=True)
    z1 = Variable(torch.zeros(1, 64, 10, 10).cuda() + 3, requires_grad=True)
    out = ca(x, y, z)
    out1 = ca1(x1, y1, z1)
    out.sum().backward()
    out1.sum().backward()
    print(x.grad,y.grad,z.grad)
    print(x1.grad,y1.grad,z1.grad)
    print('\n')
    print("%.10f" %z1.grad[0,0,0,0],"%.10f" %z.grad[0,0,0,0])
    print('\n')
    print (out)
    print (out1)
    print (abs(out1-out).sum())
    x1 = torch.randn(1, 8, 10, 10).cuda()
    y1 = torch.randn(1, 8, 10, 10).cuda()
    z1 = torch.randn(1, 64, 10, 10).cuda()
    O1 = ca1(x1, y1, z1)
    O = ca(x1, y1, z1)
    print (abs(O1-O).sum())