import torch
import torch.nn as nn
from loss import batch_episym
from torch.autograd import Variable
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx



def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_pts = x.size(2)
    x = x.view(batch_size, -1, num_pts) 
    if idx is None:
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_pts 

    idx = idx_out + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    # 4 2000 64
    x = x.transpose(2, 1).contiguous()  
    feature = x.view(batch_size * num_pts, -1)[idx, :]
    feature = feature.view(batch_size, num_pts, k, num_dims) 

    x = x.view(batch_size, num_pts, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



###########################################################################################
class PAM_Module(nn.Module):#
    def __init__(self,in_dim):#128 ,64
        nn.Module.__init__(self)
        self.query=nn.Conv2d(in_channels=in_dim,out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.softmax=nn.Softmax (dim=-1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        b,c,h,w=x.size()
        q=self.query(x).view(b,-1,w*h)
        q=q.permute(0,2,1)
        k=self.key(x).view(b,-1,w*h)
        energy=torch.bmm(q,k)
        attention=self.softmax(energy)
        v = self.value(x).view(b, -1, w * h)
        out=torch.bmm(v,attention.permute(0,2,1))
        out=out.view(b,c,h,w)
        out=self.gamma1 *out+x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM_Module, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)
        energy_new = energy_new[0].expand_as(energy)
        energy_new = energy_new - energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out




class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)




class transformer(nn.Module):
    def __init__(self, in_channel,out_channels):#channels=128
        nn.Module.__init__(self)
        self.att1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel, eps=1e-3),
            nn.BatchNorm2d(in_channel),
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channels, kernel_size=1),
        )
        self.attq1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attk1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.attv1 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.att2 = ResNet_Block(in_channel,out_channels,pre=True)
        self.gamma1 = nn.Parameter(torch.ones(1))

    def forward(self, x_row, x_local):
        x_local = self.att1(x_local) 
        q = self.attq1(x_row)  
        k = self.attk1(x_local) 
        v = self.attv1(x_local)  
        att = torch.mul(q, k)
        att = torch.softmax(att, dim=3)  
        qv = torch.mul(att, v) 
        out_local = torch.sum(qv, dim=3).unsqueeze(3)  
        out_local=torch.cat((x_row,out_local),dim=1)
        out_local=self.att2(out_local)
        out = x_row + self.gamma1 * out_local  

        return (out+out.mean(dim=1,keepdim=True))*0.5



class MLPs(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                )
    def forward(self,x):
        out = self.conv(x)
        return out

class ContextNorm(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out



class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)

class OAFilter(nn.Module):
    def __init__(self, channels, points, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points),
                nn.ReLU(),
                nn.Conv2d(points, points, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
            )
    def forward(self, x):
        out = self.conv1(x)
        y=self.conv2(out)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out  

class OAFilterBottleneck(nn.Module):
    def __init__(self, channels, points1, points2, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
        self.conv1 = nn.Sequential(
                nn.InstanceNorm2d(channels, eps=1e-3),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, out_channels, kernel_size=1),#b*c*n*1
                trans(1,2))
        self.conv2 = nn.Sequential(
                nn.BatchNorm2d(points1),
                nn.ReLU(),
                nn.Conv2d(points1, points2, kernel_size=1),
                nn.BatchNorm2d(points2),
                nn.ReLU(),
                nn.Conv2d(points2, points1, kernel_size=1)
                )
        self.conv3 = nn.Sequential(        
                trans(1,2),
                nn.InstanceNorm2d(out_channels, eps=1e-3),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=1)
                )
    def forward(self, x):
        out = self.conv1(x)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out

class diff_pool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential( 
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x):
        embed = self.conv(x)
        S = torch.softmax(embed, dim=2).squeeze(3)
        out = torch.matmul(x.squeeze(3), S.transpose(1,2)).unsqueeze(3)
        return out

class diff_unpool(nn.Module):
    def __init__(self, in_channel, output_points):
        nn.Module.__init__(self)
        self.output_points = output_points
        self.conv = nn.Sequential(
                nn.InstanceNorm2d(in_channel, eps=1e-3),
                nn.BatchNorm2d(in_channel),
                nn.ReLU(),
                nn.Conv2d(in_channel, output_points, kernel_size=1))
        
    def forward(self, x_up, x_down):

        embed = self.conv(x_up)
        S = torch.softmax(embed, dim=1).squeeze(3)
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out


class prit(nn.Module):
    def __init__(self, in_channel):
        nn.Module.__init__(self)
        self.output_points = in_channel
        self.res = ResNet_Block(in_channel, in_channel, pre=False)
        self.res2 = ResNet_Block(in_channel//2, in_channel//2, pre=False)
        self.conv1 = nn.Sequential(
            nn.InstanceNorm2d(in_channel//2, eps=1e-3),
            nn.BatchNorm2d(in_channel//2),
            nn.Conv2d(in_channel//2, in_channel//4, kernel_size=1),
            nn.ReLU())
        self.conv11 = nn.Sequential(
            nn.InstanceNorm2d(in_channel // 2, eps=1e-3),
            nn.BatchNorm2d(in_channel // 2),
            nn.Conv2d(in_channel // 2, in_channel // 4, kernel_size=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.InstanceNorm2d(in_channel//4, eps=1e-3),
            nn.BatchNorm2d(in_channel//4),
            nn.Conv2d(in_channel//4, in_channel // 8, kernel_size=1),
            nn.ReLU())
        self.conv22 = nn.Sequential(
            nn.InstanceNorm2d(in_channel // 4, eps=1e-3),
            nn.BatchNorm2d(in_channel // 4),
            nn.Conv2d(in_channel // 4, in_channel // 8, kernel_size=1),
            nn.ReLU())
        self.conv3=nn.Conv2d(in_channel // 4, 1, kernel_size=1),

    def forward(self,x):

        data=self.res(x)
        a = data[:, :64, :, :]
        b = data[:, 64:, :, :]
        data1 = self.conv1(a)
        data11 = self.conv11(b)

        c= torch.cat((data1, data11), dim=1)

        d = c[:, :32, :, :]  
        e = c[:, 32:, :, :]  

        data2 = self.conv2(d)  
        data22 = self.conv22(e)  

        f = torch.cat((data2, data22), dim=1)


        return f


class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.relu1 = nn.ReLU()
        self.left2 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.relu2 = nn.ReLU()
        self.left3 = nn.Sequential(
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
        self.relu3 = nn.ReLU()
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out1 = self.left1(x)
        out1 = self.relu1(out1) + x1
        out2 = self.left2(out1)
        out2 = self.relu2(out2) + out1 + x1
        out3 = self.left3(out2)
        out3 = self.relu3(out3 + x1)
        return out3
class MS2DGBlock(nn.Module):
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        channels = net_channels
        self.layer_num = depth
        self.k = 20
        self.head = 1
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num) + ',k:'+ str(self.k))
        self.conv1 = nn.Conv2d(input_channel, channels // 2, kernel_size=1)
        self.pam=PAM_Module(channels//2)
        self.cam=CAM_Module()
        self.pcf1=MLPs(channels//2,channels//2)
        self.pcf2=MLPs(channels//2,channels//2)

        self.att1_1 = transformer(channels,channels//2 )#
        self.att1_2 = transformer(channels, channels // 2)  #
        self.att1_3 = transformer(channels, channels // 2)  #

        l2_nums = clusters

        self.l1_1 = []
        self.l1_1.append(ContextNorm((2* channels) , channels))
        for _ in range(self.layer_num//2 - 1):
            self.l1_1.append(ContextNorm(channels))

        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        self.l1_2.append(ContextNorm(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            self.l1_2.append(ContextNorm(channels))

        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)
        self.embed_2 = prit(channels)

        self.output = nn.Conv2d(channels//4, 1, kernel_size=1)

    def forward(self, data, xs):
        n_heads=4
        batch_size, num_pts = data.shape[0], data.shape[2]
        x = self.conv1(data)
        x_pam = self.pam(x)
        x_cam = self.cam(x)
        x = self.pcf1(x_pam) + self.pcf2(x_cam)

        x_att1 = get_graph_feature(x, k=self.k)  
        x_SDG1 = self.att1_1(x, x_att1) 

        x_f = get_graph_feature(x_SDG1, k=self.k)  
        x_SDG2 = self.att1_2(x_SDG1, x_f) 

        x_g = get_graph_feature(x_SDG2, k=self.k) 
        x_SDG3 = self.att1_3(x_SDG2, x_g)  

        x = torch.cat((x, x_SDG1, x_SDG2, x_SDG3), dim=1)  



        x1_1 = self.l1_1(x)
        x_down = self.down1(x1_1)
        x2 = self.l2(x_down)

        x_up = self.up1(x1_1, x2)
        out = self.l1_2( torch.cat([x1_1,x_up], dim=1))

        out = self.embed_2(out)  
        logits = torch.squeeze(torch.squeeze(self.output(out),3),1)
        e_hat = weighted_8points(xs, logits)

        x1, x2 = xs[:,0,:,:2], xs[:,0,:,2:4]
        e_hat_norm = e_hat

        residual = batch_episym(x1, x2, e_hat_norm).reshape(batch_size, 1, num_pts, 1)

        return logits, e_hat, residual


class MS2DNET(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        self.gamma = nn.Parameter(torch.zeros(1))
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)#0
        self.weights_init = MS2DGBlock(config.net_channels, 4+self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [MS2DGBlock(config.net_channels, 6+self.side_channel, depth_each_stage, config.clusters) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        

    def forward(self, data):
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        input = data['xs'].transpose(1,3)
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        res_logits.append(logits), res_e_hat.append(e_hat)
        logits_1 = logits
        logits_2 = torch.zeros(logits.shape).cuda()

        index_k = logits.topk(k=num_pts // 2, dim=-1)[1]
        input_new = torch.stack(
            [input[i].squeeze().transpose(0, 1)[index_k[i]] for i in range(input.size(0))]).unsqueeze(-1).transpose(1,
                                                                                                                    2)

        residual_new = torch.stack(
            [residual[i].squeeze(0)[index_k[i]] for i in range(residual.size(0))]).unsqueeze(1)

        logits_new = logits.reshape(residual.shape)
        logits_new = torch.stack(
            [logits_new[i].squeeze(0)[index_k[i]] for i in range(logits_new.size(0))]).unsqueeze(1)

        data_new = torch.stack(
            [data['xs'][i].squeeze(0)[index_k[i]] for i in range(input.size(0))]).unsqueeze(1)

        for i in range(self.iter_num):
            logits, e_hat, residual = self.weights_iter[i](
                torch.cat([input_new, residual_new.detach(), torch.relu(torch.tanh(logits_new)).detach()], dim=1),
                data_new)
            logits_2.scatter_(1, index_k, logits)

            logits_2 = logits_2 + self.gamma * logits_1
            e_hat = weighted_8points(data['xs'], logits_2)
            res_logits.append(logits_2), res_e_hat.append(e_hat)

        return res_logits, res_e_hat  


        
def batch_symeig(X):
    # it is much faster to run symeig on CPU
    X = X.cpu()
    b, d, _ = X.size()
    bv = X.new(b,d,d)
    for batch_idx in range(X.shape[0]):
        # DEbuge
        # e,v = torch.symeig(X[batch_idx,:,:].squeeze(), True)
        e, v = torch.linalg.eigh(X[batch_idx, :, :].squeeze(), UPLO='U')
        bv[batch_idx,:,:] = v
    bv = bv.cuda()
    return bv


def weighted_8points(x_in, logits):
    # x_in: batch * 1 * N * 4
    x_shp = x_in.shape
    # Turn into weights for each sample
    weights = torch.relu(torch.tanh(logits))
    x_in = x_in.squeeze(1)
    
    # Make input data (num_img_pair x num_corr x 4)
    xx = torch.reshape(x_in, (x_shp[0], x_shp[2], 4)).permute(0, 2, 1)

    # Create the matrix to be used for the eight-point algorithm
    X = torch.stack([
        xx[:, 2] * xx[:, 0], xx[:, 2] * xx[:, 1], xx[:, 2],
        xx[:, 3] * xx[:, 0], xx[:, 3] * xx[:, 1], xx[:, 3],
        xx[:, 0], xx[:, 1], torch.ones_like(xx[:, 0])
    ], dim=1).permute(0, 2, 1)
    wX = torch.reshape(weights, (x_shp[0], x_shp[2], 1)) * X
    XwX = torch.matmul(X.permute(0, 2, 1), wX)

    # Recover essential matrix from self-adjoing eigen
    v = batch_symeig(XwX)
    e_hat = torch.reshape(v[:, :, 0], (x_shp[0], 9))

    # Make unit norm just in case
    e_hat = e_hat / torch.norm(e_hat, dim=1, keepdim=True)
    return e_hat

