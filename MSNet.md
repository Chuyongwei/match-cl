## MS2DNET

### 主体函数

```python
class MS2DNET(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.iter_num = config.iter_num
        self.gamma = nn.Parameter(torch.zeros(1))
        # 12//(1+1)=8
        depth_each_stage = config.net_depth//(config.iter_num+1)
        self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)#0
        # 128 4 8 500
        self.weights_init = MS2DGBlock(config.net_channels, 4+self.side_channel, depth_each_stage, config.clusters)
        self.weights_iter = [MS2DGBlock(config.net_channels, 6+self.side_channel, depth_each_stage, config.clusters) for _ in range(config.iter_num)]
        self.weights_iter = nn.Sequential(*self.weights_iter)
        
    def forward(self, data):
        # 判断是否为4维，通道数为1
        assert data['xs'].dim() == 4 and data['xs'].shape[1] == 1
        # 显示批次和样本数目
        batch_size, num_pts = data['xs'].shape[0], data['xs'].shape[2]
        # 反转1，3即 B,C,N,X->B,X,N,C 
        # C:1 N:2000 X:4
        input = data['xs'].transpose(1,3)
        # TODO 待进行 self.side_channel = (config.use_ratio==2) + (config.use_mutual==2)#0
        if self.side_channel > 0:
            sides = data['sides'].transpose(1,2).unsqueeze(3)
            input = torch.cat([input, sides], dim=1)

        res_logits, res_e_hat = [], []
        # MS2DGBlock
        # 返回估计值
        logits, e_hat, residual = self.weights_init(input, data['xs'])
        # 存储
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
```

### MS2DGBlock

```python
class MS2DGBlock(nn.Module):
    # 128 4 8 500
    def __init__(self, net_channels, input_channel, depth, clusters):
        nn.Module.__init__(self)
        # 128
        channels = net_channels
        self.layer_num = depth
        self.k = 20
        self.head = 1
        print('channels:'+str(channels)+', layer_num:'+str(self.layer_num) + ',k:'+ str(self.k))
        # 4 128//2=64 1
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
            
        # down的diff_pool
        # 128 4
        self.down1 = diff_pool(channels, l2_nums)

        self.l2 = []
        for _ in range(self.layer_num//2):
            self.l2.append(OAFilter(channels, l2_nums))

        self.up1 = diff_unpool(channels, l2_nums)

        self.l1_2 = []
        # 128*2 128
        self.l1_2.append(ContextNorm(2*channels, channels))
        for _ in range(self.layer_num//2-1):
            # 
            self.l1_2.append(ContextNorm(channels))

        # 
        self.l1_1 = nn.Sequential(*self.l1_1)
        self.l1_2 = nn.Sequential(*self.l1_2)
        self.l2 = nn.Sequential(*self.l2)
        self.embed_2 = prit(channels)

        self.output = nn.Conv2d(channels//4, 1, kernel_size=1)

    def forward(self, data, xs):
        # data input
        # xs data["xs"]
        n_heads=4
        batch_size, num_pts = data.shape[0], data.shape[2]
        # 4 4 2000 1=》4 64 2000 1
        x = self.conv1(data)
        # 卷积的结果放入pam和cam
        x_pam = self.pam(x)
        x_cam = self.cam(x)
        # 4 64 2000 1
        x = self.pcf1(x_pam) + self.pcf2(x_cam)
        
        # 处理
        # 
        x_att1 = get_graph_feature(x, k=self.k)  
        x_SDG1 = self.att1_1(x, x_att1) 

        x_f = get_graph_feature(x_SDG1, k=self.k)  
        x_SDG2 = self.att1_2(x_SDG1, x_f) 

        x_g = get_graph_feature(x_SDG2, k=self.k) 
        x_SDG3 = self.att1_3(x_SDG2, x_g)  

        x = torch.cat((x, x_SDG1, x_SDG2, x_SDG3), dim=1)  
        
        # 
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
```



### PAM_Module

```python
class PAM_Module(nn.Module):#
    def __init__(self,in_dim):#64
        nn.Module.__init__(self)
        # 64 8
        self.query=nn.Conv2d(in_channels=in_dim,out_channels=in_dim // 8, kernel_size=1)
        self.key = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels=in_dim, out_channels=in_dim , kernel_size=1)
        self.softmax=nn.Softmax (dim=-1)
        #TODO tensor拓展
        self.gamma1 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 4 64 2000 1
        b,c,h,w=x.size()
        # 4 8 2000
        q=self.query(x).view(b,-1,w*h)
        # 4 2000 8
        q=q.permute(0,2,1)
        # 4 8 2000
        k=self.key(x).view(b,-1,w*h)
        # 矩阵乘法 q(4 2000 8) x k(4 8 2000)=(4 2000 2000)
        energy=torch.bmm(q,k)
        # 针对第四维度归一化
        # 4 2000 2000
        attention=self.softmax(energy)
        # 4 64 2000
        v = self.value(x).view(b, -1, w * h)
        # 矩阵乘法 v(4 64 2000) x attention(4 2000 2000) = (4 64 2000)
        out=torch.bmm(v,attention.permute(0,2,1))
        
        # (4 64 1 2000)
        out=out.view(b,c,h,w)
        # 
        out=self.gamma1 *out+x
        return out
```

### CAM_Module

```python
class CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self):
        super(CAM_Module, self).__init__()

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # 4 64 2000 1
        m_batchsize, C, height, width = x.size()
        # 4 64 2000
        proj_query = x.view(m_batchsize, C, -1)
        # 4 2000 64
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        # 自乘 4 64 64
        energy = torch.bmm(proj_query, proj_key)
        # 获取每个组中最大值 4 64 1
        energy_new = torch.max(energy, -1, keepdim=True)
        # 扩展函数
        energy_new = energy_new[0].expand_as(energy)
        # 获取差值
        energy_new = energy_new - energy
        # 4 64 64
        attention = self.softmax(energy_new)
        # 4 64 2000
        proj_value = x.view(m_batchsize, C, -1)
        # 4 64 2000
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out
```

### MLPs

多层处理

```python
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
```

### get_graph_feature

获得图的特征

```py
# 获取数据
def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    # 用于从张量中提取前 K 个最大（或最小）的元素及其对应的索引。它在处理排序、选择最大值或最小值以及生成推荐系统等任务中非常有用
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  
    return idx

# 获得图特征

def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_pts = x.size(2)
    x = x.view(batch_size, -1, num_pts) 
    if idx is None:
        # 4 2000 20
        idx_out = knn(x, k=k)
    else:
        idx_out = idx
    device = torch.device('cuda')

    # （4 1 1）0 2000 4000 6000
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_pts 

    # 4 2000 20
    idx = idx_out + idx_base

    # 160000
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  
    feature = x.view(batch_size * num_pts, -1)[idx, :]
    feature = feature.view(batch_size, num_pts, k, num_dims) 

    x = x.view(batch_size, num_pts, 1, num_dims).repeat(1, 1, k, 1) 
    feature = torch.cat((x, x - feature), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature
```

### transformer

转换

```python
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
        # 4 64 2000 1 
        # 4 64 2000 20
        x_local = self.att1(x_local) 
        
        q = self.attq1(x_row)  
        k = self.attk1(x_local) 
        v = self.attv1(x_local)  
        att = torch.mul(q, k)
        att = torch.softmax(att, dim=3)  
        #4 64 2000 20
        qv = torch.mul(att, v) 
        out_local = torch.sum(qv, dim=3).unsqueeze(3)  
        out_local=torch.cat((x_row,out_local),dim=1)
        out_local=self.att2(out_local)
        #4 64 2000 1
        out = x_row + self.gamma1 * out_local  

        return (out+out.mean(dim=1,keepdim=True))*0.5
```

#### 残差网络



```python
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
```

### diff_pool和diff_unpool



```python
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
        # 4 128 2000 1
        # 4 500 2000 1
        embed = self.conv(x)
        # 取得极大值
        S = torch.softmax(embed, dim=2).squeeze(3)
        # x和S相乘
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
        # 卷积
        embed = self.conv(x_up)
        # 提取
        S = torch.softmax(embed, dim=1).squeeze(3)
        # x——down和S相乘
        out = torch.matmul(x_down.squeeze(3), S).unsqueeze(3)
        return out
```

### ContextNorm

> ContextNorm 是一种新型的神经网络归一化方法，旨在通过利用上下文信息来提高模型的收敛速度和性能。与传统的 Batch Normalization 和 Group Normalization 不同，ContextNorm 通过将数据分组为不同的“上下文”，并在每个上下文中独立进行归一化，从而更好地适应动态激活分布。

#### ContextNorm 的核心思想

- ContextNorm 引入了基于领域知识的上下文（context）概念，将数据划分为不同的上下文组。
- 在每个上下文中，ContextNorm 会计算均值和方差，并对数据进行归一化。
- 在反向传播过程中，ContextNorm 会为每个上下文学习归一化参数和模型权重，从而实现高效的收敛。

#### ContextNorm 的优势

1. **适应性强**：ContextNorm 不依赖于全局的均值和方差，而是根据上下文进行局部归一化，更适合处理动态激活分布。
2. **性能提升**：实验表明，ContextNorm 在多种任务（如分类和域适应）中表现优于 Batch Normalization 和 Mixture Normalization。
3. **灵活性高**：ContextNorm 可以作为独立的层插入到任何神经网络架构中，并且可以在网络的任何层级中使用。

#### 应用场景

- **分类任务**：在 CIFAR-10、CIFAR-100 和 Tiny ImageNet 等数据集上，ContextNorm 显著提高了模型的收敛速度和最终性能。
- **域适应**：在源域和目标域之间使用 ContextNorm，可以更好地适应不同域的分布差异。
- **超类别分类**：例如在 CIFAR-100 的超类别分类任务中，ContextNorm 提升了模型的分类性能

```python
class ContextNorm(nn.Module):
    def __init__(self, channels, out_channels=None):
        nn.Module.__init__(self)
        if not out_channels:
           # 2*128 = 256
           out_channels = channels
        self.shot_cut = None
        if out_channels != channels:
            self.shot_cut = nn.Conv2d(channels, out_channels, kernel_size=1)
            # TODO:需要研究
        self.conv = nn.Sequential(
            #256
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
        # 4 256 2000 1
        out = self.conv(x)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            # 4 128 2000 1
            out = out + x
        return out
```

### OAFilter

```python
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
        # 4 128 500 1
        # 4 500 128 1
        out = self.conv1(x)
        y=self.conv2(out)
        out = out + self.conv2(out)
        out = self.conv3(out)
        if self.shot_cut:
            out = out + self.shot_cut(x)
        else:
            out = out + x
        return out  
```

#### trans

> 交换1和2

```python
class trans(nn.Module):
    def __init__(self, dim1, dim2):
        nn.Module.__init__(self)
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)
```

```python
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
```

