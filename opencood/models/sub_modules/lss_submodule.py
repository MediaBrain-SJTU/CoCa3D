from re import A
import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import torch.nn.functional as F
from opencood.utils.camera_utils import bin_depths
from opencood.models.fuse_modules.att_fuse import ScaledDotProductAttention
from opencood.models.sub_modules.torch_transformation_utils import \
    warp_affine_simple
from opencood.models.fuse_modules.fuse_utils import regroup as Regroup

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)  # 上采样 BxCxHxW->BxCx2Hx2W

        self.conv = nn.Sequential(  # 两个3x3卷积
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),  # inplace=True使用原地操作，节省内存
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 对x1进行上采样
        x1 = torch.cat([x2, x1], dim=1)  # 将x1和x2 concat 在一起
        return self.conv(x1)


class CamEncode(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample, ddiscr, mode, use_gt_depth=False, depth_supervision=True):
        super(CamEncode, self).__init__()
        self.D = D  # 42
        self.C = C  # 64
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode
        self.use_gt_depth = use_gt_depth
        self.depth_supervision = depth_supervision # in the case of not use gt depth

        
        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征

        self.up1 = Up(320+112, 512)  # 上采样模块，输入输出通道分别为320+112和512
        if downsample == 8:
            self.up2 = Up(512+40, 512)
        if not use_gt_depth:
            self.depth_head = nn.Conv2d(512, self.D, kernel_size=1, padding=0)  # 1x1卷积，变换维度
        self.image_head = nn.Conv2d(512, self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-5):  # 对深度维进行softmax，得到每个像素不同深度的概率
        return F.softmax(x, dim=1)

    def get_gt_depth_dist(self, x):  # 对深度维进行onehot，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        target = self.training
        torch.clamp_max_(x, self.d_max) # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices, mask = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=target)
        depth_indices = depth_indices[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample]
        onehot_dist = F.one_hot(depth_indices.long()).permute(0,3,1,2) # [B*N, num_bins, fH, fW]

        if not target:
            mask = mask[:, self.downsample//2::self.downsample, self.downsample//2::self.downsample].unsqueeze(1)
            onehot_dist *= mask

        return onehot_dist, depth_indices

    def get_eff_features(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231

        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x  # x: 24 x 320 x 4 x 11
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        if self.downsample == 8:
            x = self.up2(x, endpoints['reduction_3'])
        return x  # x: 24 x 512 x 8 x 22


    def forward(self, x):
        """
        Returns:
            log_depth : [B*N, D, fH, fW], or None if not used latter
            depth_gt_indices : [B*N, fH, fW], or None if not used latter
            new_x : [B*N, C, D, fH, fW]
        """
        x_img = x[:,:3:,:,:]
        x_depth = x[:,3,:,:]
        features = self.get_eff_features(x_img)  
        x_img = self.image_head(features) #  x: B*N x C x fH x fW(24 x 64 x 8 x 22)

        depth_gt, depth_gt_indices = self.get_gt_depth_dist(x_depth)
        if self.use_gt_depth:
            new_x = depth_gt.unsqueeze(1) * x_img.unsqueeze(2) # new_x: 24 x 64 x 41 x 8 x 18
            return None, None, depth_gt, x_img, new_x
        else:
            depth_logit = self.depth_head(features)
            depth = self.get_depth_dist(depth_logit)
            new_x = depth.unsqueeze(1) * x_img.unsqueeze(2) # new_x: 24 x 64 x 41 x 8 x 18
            if self.depth_supervision:
                return depth_logit, depth_gt_indices, depth_gt, x_img, new_x
            else:
                return None, None, depth_gt, x_img, new_x


class CamEncodeGTDepth(nn.Module):  # 提取图像特征进行图像编码
    def __init__(self, D, C, downsample, ddiscr, mode):
        """
        Args:
            downsample : image feature downsample ratio
            ddiscr : [d_min, d_max, num_bins]
            mode : "UD" uniform discretization or "LID" Linear interval discretization
        """
        super(CamEncodeGTDepth, self).__init__()
        self.C = C  # 64
        self.D = D  # 42
        self.downsample = downsample
        self.d_min = ddiscr[0]
        self.d_max = ddiscr[1]
        self.num_bins = ddiscr[2]
        self.mode = mode

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")  # 使用 efficientnet 提取特征

        self.up1 = Up(320+112, 512)  # 上采样模块，输入输出通道分别为320+112和512
        if downsample == 8:
            self.up2 = Up(512+40, 512)
        self.net = nn.Conv2d(512, self.C, kernel_size=1, padding=0)  # 1x1卷积，变换维度

    def get_depth_dist(self, x, soft=False):  # 对深度维进行softmax，得到每个像素不同深度的概率
        """
        Args:
            x: [B*N, H, W]
        Returns:
            x: [B*N, D, fH, fW]
        """
        torch.clamp_max_(x, self.d_max+5) # save memory
        # [B*N, H, W], indices (float), value: [0, num_bins)
        depth_indices = bin_depths(x, self.mode, self.d_min, self.d_max, self.num_bins, target=False)
        onehot_dist = F.one_hot(depth_indices.long()).permute(0,3,1,2) # [B*N, num_bins, H, W]
        depth_dist = onehot_dist[:,:, \
                                 self.downsample//2::self.downsample, self.downsample//2::self.downsample] # [B*N,num_bins,fH,fW]
        return depth_dist


    def get_depth_feat(self, x):
        x_img = x[:,:3:,:,:]
        x_depth = x[:,3,:,:]

        x_img = self.get_eff_feature(x_img)  # 使用efficientnet提取特征  x: 24 x 512 x 8 x 18
        # Depth
        x_img = self.net(x_img)  # 1x1卷积变换维度  x: 24 x 64(C) x 8 x 18

        depth = self.get_depth_dist(x_depth)  # 第二个维度的前D个作为深度维，进行softmax  depth: 24 x 41 x 8 x 18
        new_x = depth.unsqueeze(1) * x_img.unsqueeze(2)  # 将特征通道维和通道维利用广播机制相乘  new_x: 24 x 64 x 41 x 8 x 18

        return depth, new_x

    def get_eff_feature(self, x):  # 使用efficientnet提取特征
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))  #  x: 24 x 32 x 64 x 176
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks) # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints)+1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints)+1)] = x  # x: 24 x 320 x 4 x 11
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])  # 先对endpoints[4]进行上采样，然后将 endpoints[5]和endpoints[4] concat 在一起
        if self.downsample == 8:
            x = self.up2(x, endpoints['reduction_3'])
        return x  # x: 24 x 512 x 8 x 22

    def forward(self, x):
        depth, x = self.get_depth_feat(x)  # depth: B*N x D x fH x fW(24 x 41 x 8 x 22)  x: B*N x C x D x fH x fW(24 x 64 x 41 x 8 x 22)

        return depth, x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):  # inC: 64  outC: not 1 for object detection
        super(BevEncode, self).__init__()

        # 使用resnet的前3个stage作为backbone
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up1 = Up(64+256, 256, scale_factor=4)
        self.up2 = nn.Sequential(  # 2倍上采样->3x3卷积->1x1卷积
            nn.Upsample(scale_factor=2, mode='bilinear',
                              align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):  # x: 4 x 64 x 240 x 240
        x = self.conv1(x)  # x: 4 x 64 x 120 x 120
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: 4 x 64 x 120 x 120
        x = self.layer2(x1)  # x: 4 x 128 x 60 x 60
        x = self.layer3(x)  # x: 4 x 256 x 30 x 30

        x = self.up1(x, x1)  # 给x进行4倍上采样然后和x1 concat 在一起  x: 4 x 256 x 120 x 120
        x = self.up2(x)  # 2倍上采样->3x3卷积->1x1卷积  x: 4 x 1 x 240 x 240

        return x


class BevEncodeMSFusion(nn.Module):
    """
    Multiscale version of ResNet Encoder
    """
    def __init__(self, fusion_args):  # inC: 64  outC: not 1 for object detection
        super(BevEncodeMSFusion, self).__init__()
        args = fusion_args['args']
        inC = args['in_channels']
        self.discrete_ratio = args['voxel_size'][0]  
        self.downsample_rate = 1
        # 使用resnet的前3个stage作为backbone
        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu # make it 64 channels

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        self.up_layer1 = Up(64+256, 256, scale_factor=2)
        self.up_layer2 = Up(128+256, 256, scale_factor=2)
        self.down_layer = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3,
                      stride=1,padding=1),
            nn.ReLU(inplace=True)
        )
        if fusion_args['core_method'] == "max_ms":
            self.fuse_module = [MaxFusion(), MaxFusion(), MaxFusion()]
        elif fusion_args['core_method'] == "att_ms":
            self.fuse_module = [AttFusion(64), AttFusion(128), AttFusion(256)]
        else:
            raise "not implemented"

    def forward(self, x, record_len, pairwise_t_matrix):  # x: 4 x 64 x 240 x 240
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # (B,L,L,2,3)
        pairwise_t_matrix = pairwise_t_matrix[:,:,:,[0, 1],:][:,:,:,:,[0, 1, 3]] # [B, L, L, 2, 3]
        pairwise_t_matrix[...,0,1] = pairwise_t_matrix[...,0,1] * H / W
        pairwise_t_matrix[...,1,0] = pairwise_t_matrix[...,1,0] * W / H
        pairwise_t_matrix[...,0,2] = pairwise_t_matrix[...,0,2] / (self.downsample_rate * self.discrete_ratio * W) * 2
        pairwise_t_matrix[...,1,2] = pairwise_t_matrix[...,1,2] / (self.downsample_rate * self.discrete_ratio * H) * 2

        x = self.conv1(x)  # x: 4 x 64 x 120 x 120
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)  # x1: 4 x 64 x 120 x 120
        x2 = self.layer2(x1)  # x2: 4 x 128 x 60 x 60
        x3 = self.layer3(x2)  # x3: 4 x 256 x 30 x 30
        x_single = self.down_layer(self.up_layer1(self.up_layer2(x3, x2), x1)) # 4 x 64 x 120 x 120

        x1_fuse = self.fuse_module[0](x1, record_len, pairwise_t_matrix)
        x2_fuse = self.fuse_module[1](x2, record_len, pairwise_t_matrix)
        x3_fuse = self.fuse_module[2](x3, record_len, pairwise_t_matrix)

        x_fuse = self.down_layer(self.up_layer1(self.up_layer2(x3_fuse, x2_fuse), x1_fuse)) # 4 x 64 x 120 x 120

        return x_single, x_fuse

def regroup(x, record_len):
    cum_sum_len = torch.cumsum(record_len, dim=0)
    split_x = torch.tensor_split(x, cum_sum_len[:-1].cpu())
    return split_x

class MaxFusion(nn.Module):
    def __init__(self):
        super(MaxFusion, self).__init__()

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            out.append(torch.max(neighbor_feature, dim=0)[0])
        out = torch.stack(out)
        
        return out

class AttFusion(nn.Module):
    def __init__(self, feature_dims):
        super(AttFusion, self).__init__()
        self.att = ScaledDotProductAttention(feature_dims)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        batch_node_features = split_x
        out = []
        # iterate each batch
        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            # update each node i
            i = 0 # ego
            x = warp_affine_simple(batch_node_features[b], t_matrix[i, :, :, :], (H, W))
            cav_num = x.shape[0]
            x = x.view(cav_num, C, -1).permute(2, 0, 1) #  (H*W, cav_num, C), perform self attention on each pixel.
            h = self.att(x, x, x)
            h = h.permute(1, 2, 0).view(cav_num, C, H, W)[0, ...]  # C, W, H before
            out.append(h)

        out = torch.stack(out)
        return out

class DiscoFusion(nn.Module):
    def __init__(self, feature_dims):
        super(DiscoFusion, self).__init__()
        from opencood.models.fuse_modules.disco_fuse import PixelWeightLayer
        self.pixel_weight_layer = PixelWeightLayer(feature_dims)

    def forward(self, xx, record_len, pairwise_t_matrix):
        _, C, H, W = xx.shape
        B, L = pairwise_t_matrix.shape[:2]
        split_x = regroup(xx, record_len)
        out = []

        for b in range(B):
            N = record_len[b]
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]
            i = 0 # ego
            neighbor_feature = warp_affine_simple(split_x[b],
                                            t_matrix[i, :, :, :],
                                            (H, W))
            # (N, C, H, W)
            ego_feature = split_x[b][0].view(1, C, H, W).expand(N, -1, -1, -1)
            # (N, 2C, H, W)
            neighbor_feature_cat = torch.cat((neighbor_feature, ego_feature), dim=1)
            # (N, 1, H, W)
            agent_weight = self.pixel_weight_layer(neighbor_feature_cat) 
            # (N, 1, H, W)
            agent_weight = F.softmax(agent_weight, dim=0)

            agent_weight = agent_weight.expand(-1, C, -1, -1)
            # (N, C, H, W)
            feature_fused = torch.sum(agent_weight * neighbor_feature, dim=0)
            out.append(feature_fused)

        return torch.stack(out)

class V2VNetFusion(nn.Module):
    def __init__(self, args):
        super(V2VNetFusion, self).__init__()
        from opencood.models.sub_modules.convgru import ConvGRU
        in_channels = args['in_channels']
        H, W = args['conv_gru']['H'], args['conv_gru']['W'] # remember to modify for v2xsim dataset
        kernel_size = args['conv_gru']['kernel_size']
        num_gru_layers = args['conv_gru']['num_layers']
        self.num_iteration = args['num_iteration']
        self.gru_flag = args['gru_flag']
        self.agg_operator = args['agg_operator']

        self.msg_cnn = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3,
                                 stride=1, padding=1)
        self.conv_gru = ConvGRU(input_size=(H, W),
                                input_dim=in_channels * 2,
                                hidden_dim=[in_channels] * num_gru_layers,
                                kernel_size=kernel_size,
                                num_layers=num_gru_layers,
                                batch_first=True,
                                bias=True,
                                return_all_layers=False)
        self.mlp = nn.Linear(in_channels, in_channels)

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        split_x = regroup(x, record_len)
        # (B*L,L,1,H,W)
        roi_mask = torch.zeros((B, L, L, 1, H, W)).to(x)
        for b in range(B):
            N = record_len[b]
            for i in range(N):
                one_tensor = torch.ones((L,1,H,W)).to(x)
                roi_mask[b,i] = warp_affine_simple(one_tensor, pairwise_t_matrix[b][i, :, :, :],(H, W))

        batch_node_features = split_x
        # iteratively update the features for num_iteration times
        for l in range(self.num_iteration):

            batch_updated_node_features = []
            # iterate each batch
            for b in range(B):

                # number of valid agent
                N = record_len[b]
                # (N,N,4,4)
                # t_matrix[i, j]-> from i to j
                t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

                updated_node_features = []

                # update each node i
                for i in range(N):
                    # (N,1,H,W)
                    mask = roi_mask[b, i, :N, ...]
                    neighbor_feature = warp_affine_simple(batch_node_features[b],
                                                   t_matrix[i, :, :, :],
                                                   (H, W))

                    # (N,C,H,W)
                    ego_agent_feature = batch_node_features[b][i].unsqueeze(
                        0).repeat(N, 1, 1, 1)
                    #(N,2C,H,W)
                    neighbor_feature = torch.cat(
                        [neighbor_feature, ego_agent_feature], dim=1)
                    # (N,C,H,W)
                    # message contains all feature map from j to ego i.
                    message = self.msg_cnn(neighbor_feature) * mask

                    # (C,H,W)
                    if self.agg_operator=="avg":
                        agg_feature = torch.mean(message, dim=0)
                    elif self.agg_operator=="max":
                        agg_feature = torch.max(message, dim=0)[0]
                    else:
                        raise ValueError("agg_operator has wrong value")
                    # (2C, H, W)
                    cat_feature = torch.cat(
                        [batch_node_features[b][i, ...], agg_feature], dim=0)
                    # (C,H,W)
                    if self.gru_flag:
                        gru_out = \
                            self.conv_gru(cat_feature.unsqueeze(0).unsqueeze(0))[
                                0][
                                0].squeeze(0).squeeze(0)
                    else:
                        gru_out = batch_node_features[b][i, ...] + agg_feature
                    updated_node_features.append(gru_out.unsqueeze(0))
                # (N,C,H,W)
                batch_updated_node_features.append(
                    torch.cat(updated_node_features, dim=0))
            batch_node_features = batch_updated_node_features
        # (B,C,H,W)
        out = torch.cat(
            [itm[0, ...].unsqueeze(0) for itm in batch_node_features], dim=0)
        # (B,C,H,W) -> (B, H, W, C) -> (B,C,H,W)
        out = self.mlp(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return out

class V2XViTFusion(nn.Module):
    def __init__(self, args):
        super(V2XViTFusion, self).__init__()
        from opencood.models.sub_modules.v2xvit_basic import V2XTransformer
        from opencood.models.sub_modules.torch_transformation_utils import warp_affine_simple
        self.discrete_ratio = args['voxel_size'][0]  # voxel_size[0]=0.4    
        self.downsample_rate = args['downsample_rate']  # 2/4, downsample rate from original feature map [200, 704]
        self.fusion_net = V2XTransformer(args['transformer'])

    def forward(self, x, record_len, pairwise_t_matrix):
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        regroup_feature, mask = Regroup(x, record_len, L)
        prior_encoding = \
            torch.zeros(len(record_len), L, 3, 1, 1).to(record_len.device)
        
        # prior encoding added
        prior_encoding = prior_encoding.repeat(1, 1, 1,
                                               regroup_feature.shape[3],
                                               regroup_feature.shape[4])

        regroup_feature = torch.cat([regroup_feature, prior_encoding], dim=2)
        regroup_feature_new = []

        for b in range(B):
            # (B,L,L,2,3)
            ego = 0
            regroup_feature_new.append(warp_affine_simple(regroup_feature[b], pairwise_t_matrix[b, ego], (H, W)))
        regroup_feature = torch.stack(regroup_feature_new)

        # b l c h w -> b l h w c
        regroup_feature = regroup_feature.permute(0, 1, 3, 4, 2)
        # transformer fusion
        spatial_correction_matrix = torch.eye(4).expand(len(record_len), L, 4, 4).to(record_len.device)
        fused_feature = self.fusion_net(regroup_feature, mask, spatial_correction_matrix)
        # b h w c -> b c h w
        fused_feature = fused_feature.permute(0, 3, 1, 2)
        
        return fused_feature

class When2commFusion(nn.Module):
    def __init__(self, args):
        super(When2commFusion, self).__init__()
        import numpy as np
        from opencood.models.fuse_modules.when2com_fuse import policy_net4, km_generator, MIMOGeneralDotProductAttention

        self.in_channels = args['in_channels']
        self.feat_H = args['H']
        self.feat_W = args['W']
        self.query_size = args['query_size']
        self.key_size = args['key_size']
        self.mode = args['mode']
        

        self.query_key_net = policy_net4(self.in_channels*2)
        self.key_net = km_generator(out_size=self.key_size, input_feat_h=self.feat_H//4, input_feat_w=self.feat_W//4)
        self.query_net = km_generator(out_size=self.query_size, input_feat_h=self.feat_H//4, input_feat_w=self.feat_W//4)
        self.attention_net = MIMOGeneralDotProductAttention(self.query_size, self.key_size)

    def forward(self, x, record_len, pairwise_t_matrix):
        """
        Fusion forwarding.
        
        Parameters
        ----------
        x : torch.Tensor
            input data, (sum(n_cav), C, H, W)
            
        record_len : list
            shape: (B)
            
        pairwise_t_matrix : torch.Tensor
            The transformation matrix from each cav to ego, 
            shape: (B, L, L, 4, 4) 
        
        weight: torch.Tensor
            Weight of aggregating coming message
            shape: (B, L, L)
            
        Returns
        -------
        Fused feature.
        """
        _, C, H, W = x.shape
        B, L = pairwise_t_matrix.shape[:2]

        # split x:[(L1, C, H, W), (L2, C, H, W), ...]
        # for example [[2, 256, 50, 176], [1, 256, 50, 176], ...]
        split_x = regroup(x, record_len)
        batch_node_features = split_x
        updated_node_features = []
        for b in range(B):

            # number of valid agent
            N = record_len[b]
            # (N,N,4,4)
            # t_matrix[i, j]-> from i to j
            t_matrix = pairwise_t_matrix[b][:N, :N, :, :]

            # update each node i
            # (N,1,H,W)
            # (N,C,H,W) neighbor_feature is agent i's neighborhood warping to agent i's perspective
            # Notice we put i one the first dim of t_matrix. Different from original.
            # t_matrix[i,j] = Tji
            neighbor_feature = warp_affine_simple(batch_node_features[b],
                                            t_matrix[0, :, :, :],
                                            (H, W))
            query_key_maps = self.query_key_net(neighbor_feature)
            keys = self.key_net(query_key_maps)
            query = self.query_net(query_key_maps[0].unsqueeze(0))

            query = query.unsqueeze(0)
            keys = keys.unsqueeze(0)
            neighbor_feature = neighbor_feature.unsqueeze(1).unsqueeze(0)

            feat_fuse, prob_action = self.attention_net(query, keys, neighbor_feature, sparse=True)

            updated_node_features.append(feat_fuse.squeeze(0))

        out = torch.cat(updated_node_features, dim=0)
        
        return out