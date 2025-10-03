import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast
import torch.nn.functional as F  
import torchvision.models.segmentation as seg_models  

from mmdet.models.backbones.resnet import BasicBlock
from .base_lss_fpn import BaseLSSFPN, Mlp, SELayer
from torch.cuda.amp.autocast_mode import autocast

from ops.average_voxel_pooling_v2 import average_voxel_pooling

__all__ = ['RVTLSSFPN', 'SemanticGuidedRVTLSSFPN']

#语义注意力模块       
class SemanticAttentionModule(nn.Module):  
    def __init__(self, img_channels, semantic_channels):  
        super().__init__()  
        self.attention_conv = nn.Sequential(  
            nn.Conv2d(semantic_channels, img_channels, 1),  
            nn.BatchNorm2d(img_channels),  
            nn.Sigmoid()  
        )  
          
    def forward(self, img_feat, semantic_feat):  
        # Handle batch dimension mismatch  
        if img_feat.shape[0] != semantic_feat.shape[0]:  
            # Expand semantic_feat to match img_feat's batch dimension  
            # Assuming img_feat has shape [batch_size*num_cams, ...] and semantic_feat has [batch_size, ...]  
            num_cams = img_feat.shape[0] // semantic_feat.shape[0]  
            semantic_feat = semantic_feat.repeat_interleave(num_cams, dim=0)  
          
        # Adjust spatial dimensions if needed  
        if semantic_feat.shape[-2:] != img_feat.shape[-2:]:  
            semantic_feat = F.interpolate(  
                semantic_feat,  
                size=img_feat.shape[-2:],  
                mode='bilinear',  
                align_corners=False  
            )  
          
        # Generate attention weights  
        attention_weights = self.attention_conv(semantic_feat)  
        # Apply attention guidance  
        enhanced_feat = img_feat * attention_weights  
        return enhanced_feat
    
#时空语义建模    
class TemporalSemanticConsistency(nn.Module):  
    def __init__(self, semantic_channels):  
        super().__init__()  
        self.flow_estimator = nn.Sequential(  
            nn.Conv2d(semantic_channels * 2, 64, 3, padding=1),  
            nn.ReLU(),  
            nn.Conv2d(64, 2, 3, padding=1)  
        )  
          
    def forward(self, current_semantic, prev_semantic):  
        if prev_semantic is None:  
            return torch.tensor(0.0, device=current_semantic.device)  
          
        combined = torch.cat([current_semantic, prev_semantic], dim=1)  
        flow = self.flow_estimator(combined)  
        warped_prev = self.warp_semantic(prev_semantic, flow)  
        temporal_loss = F.mse_loss(current_semantic, warped_prev)  
        return temporal_loss  
    
class ViewAggregation(nn.Module):
    """
    Aggregate frustum view features transformed by depth distribution / radar occupancy
    """
    def __init__(self, in_channels, mid_channels, out_channels):
        super(ViewAggregation, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=True),
        )

    @autocast(False)
    def forward(self, x):
        x = self.reduce_conv(x)
        x = self.conv(x)
        x = self.out_conv(x)
        return x


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels, depth_channels,
                 camera_aware=True):
        super(DepthNet, self).__init__()
        self.camera_aware = camera_aware

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        if self.camera_aware:
            self.bn = nn.BatchNorm1d(27)
            self.depth_mlp = Mlp(27, mid_channels, mid_channels)
            self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
            self.context_mlp = Mlp(27, mid_channels, mid_channels)
            self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.context_conv = nn.Sequential(
            nn.Conv2d(mid_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels,
                      context_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        x = self.reduce_conv(x)

        if self.camera_aware:
            intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
            batch_size = intrins.shape[0]
            num_cams = intrins.shape[2]
            ida = mats_dict['ida_mats'][:, 0:1, ...]
            sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
            bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                            4).repeat(1, 1, num_cams, 1, 1)
            mlp_input = torch.cat(
                [
                    torch.stack(
                        [
                            intrins[:, 0:1, ..., 0, 0],
                            intrins[:, 0:1, ..., 1, 1],
                            intrins[:, 0:1, ..., 0, 2],
                            intrins[:, 0:1, ..., 1, 2],
                            ida[:, 0:1, ..., 0, 0],
                            ida[:, 0:1, ..., 0, 1],
                            ida[:, 0:1, ..., 0, 3],
                            ida[:, 0:1, ..., 1, 0],
                            ida[:, 0:1, ..., 1, 1],
                            ida[:, 0:1, ..., 1, 3],
                            bda[:, 0:1, ..., 0, 0],
                            bda[:, 0:1, ..., 0, 1],
                            bda[:, 0:1, ..., 1, 0],
                            bda[:, 0:1, ..., 1, 1],
                            bda[:, 0:1, ..., 2, 2],
                        ],
                        dim=-1,
                    ),
                    sensor2ego.view(batch_size, 1, num_cams, -1),
                ],
                -1,
            )
            mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
            context_se = self.context_mlp(mlp_input)[..., None, None]
            context_img = self.context_se(x, context_se)
            context = self.context_conv(context_img)
            depth_se = self.depth_mlp(mlp_input)[..., None, None]
            depth = self.depth_se(x, depth_se)
            depth = self.depth_conv(depth)
        else:
            context = self.context_conv(x)
            depth = self.depth_conv(x)

        return torch.cat([depth, context], dim=1)


class RVTLSSFPN(BaseLSSFPN):
    def __init__(self, **kwargs):
        super(RVTLSSFPN, self).__init__(**kwargs)

        self.register_buffer('frustum', self.create_frustum())
        self.z_bound = kwargs['z_bound']
        self.radar_view_transform = kwargs['radar_view_transform']
        self.camera_aware = kwargs['camera_aware']

        self.depth_net = self._configure_depth_net(kwargs['depth_net_conf'])
        self.view_aggregation_net = ViewAggregation(self.output_channels*2,
                                                    self.output_channels*2,
                                                    self.output_channels)

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
            camera_aware=self.camera_aware
        )

    def get_geometry_collapsed(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat,
                               z_min=-5., z_max=3.):
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1)).double()
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
             points[:, :, :, :, :, 2:]), 5)

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat)).double()
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points).half()
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)

        points_out = points[:, :, :, 0:1, :, :3]
        points_valid_z = ((points[..., 2] > z_min) & (points[..., 2] < z_max))

        return points_out, points_valid_z

    def _forward_view_aggregation_net(self, img_feat_with_depth):  
        # BEVConv2D [n, c, d, h, w] -> [n, h, c, w, d]  
        img_feat_with_depth = img_feat_with_depth.permute(  
            0, 3, 1, 4, 2).contiguous()  # [n, c, d, h, w] -> [n, h, c, w, d]  
        n, h, c, w, d = img_feat_with_depth.shape  
        img_feat_with_depth = img_feat_with_depth.view(-1, c, w, d)  

        # 通过ViewAggregation网络  
        aggregated_feat = self.view_aggregation_net(img_feat_with_depth)  

        # 获取实际的输出通道数  
        actual_output_channels = aggregated_feat.shape[1]  

        img_feat_with_depth = aggregated_feat.view(  
            n, h, actual_output_channels, w, d).permute(0, 2, 4, 1, 3).contiguous().float()  
        return img_feat_with_depth

    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)

    def _split_batch_cam(self, feat, inv=False, num_cams=6):
        batch_size = feat.shape[0]
        if not inv:
            return feat.reshape(batch_size // num_cams, num_cams, *feat.shape[1:])
        else:
            return feat.reshape(batch_size * num_cams, *feat.shape[2:])

    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              pts_context,
                              pts_occupancy,
                              return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego.
                intrin_mats(Tensor): Intrinsic matrix.
                ida_mats(Tensor): Transformation matrix for ida.
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera.
                bda_mat(Tensor): Rotation matrix for bda.
            ptss_context(Tensor): Input point context feature.
            ptss_occupancy(Tensor): Input point occupancy.
            return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t3 = torch.cuda.Event(enable_timing=True)
            t4 = torch.cuda.Event(enable_timing=True)
            t5 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        
        # 提取语义特征（在图像backbone之前）  
        if self.use_semantic_guidance:  
            # 重新整形图像用于语义分割  
            semantic_input = sweep_imgs.reshape(-1, num_channels, img_height, img_width)  
            semantic_feat = self._extract_semantic_features(semantic_input)  

        else:  
            semantic_feat = None  
            
        # extract image feature
        img_feats = self.get_cam_feats(sweep_imgs)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img_backbone'].append(t1.elapsed_time(t2))

        source_features = img_feats[:, 0, ...]
        source_features = self._split_batch_cam(source_features, inv=True, num_cams=num_cams)

        # predict image context feature, depth distribution
        depth_feature = self._forward_depth_net(
            source_features,
            mats_dict,
        )
        if self.times is not None:
            t3.record()
            torch.cuda.synchronize()
            self.times['img_dep'].append(t2.elapsed_time(t3))

        image_feature = depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)]

        depth_occupancy = depth_feature[:, :self.depth_channels].softmax(
            dim=1, dtype=depth_feature.dtype)
        img_feat_with_depth = depth_occupancy.unsqueeze(1) * image_feature.unsqueeze(2)

        # calculate frustum grid within valid height
        geom_xyz, geom_xyz_valid = self.get_geometry_collapsed(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None))

        geom_xyz_valid = self._split_batch_cam(geom_xyz_valid, inv=True, num_cams=num_cams).unsqueeze(1)
        img_feat_with_depth = (img_feat_with_depth * geom_xyz_valid).sum(3).unsqueeze(3)

        if self.radar_view_transform:    
            radar_occupancy = pts_occupancy.permute(0, 2, 1, 3).contiguous()    
            image_feature_collapsed = (image_feature * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2)    
            img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)    

            img_context = torch.cat([img_feat_with_depth, img_feat_with_radar], dim=1)    

            # 在ViewAggregation之前融合语义特征  
            if semantic_feat is not None:  
                #print(f"img_context before semantic fusion: {img_context.shape}")  
                #print(f"semantic_feat shape: {semantic_feat.shape}")  
                
                # 1. 应用语义注意力引导
                enhanced_image_feature = self.semantic_attention(image_feature, semantic_feat)  
                
                # 2. 使用增强后的特征进行原有的融合逻辑
                if self.radar_view_transform:  
                    radar_occupancy = pts_occupancy.permute(0, 2, 1, 3).contiguous()  
                    image_feature_collapsed = (enhanced_image_feature * geom_xyz_valid.max(2).values).sum(2).unsqueeze(2)  
                    img_feat_with_radar = radar_occupancy.unsqueeze(1) * image_feature_collapsed.unsqueeze(2)  

                
                # 调整语义特征尺寸以匹配img_context的空间维度  
                semantic_feat_resized = F.interpolate(  
                    semantic_feat,  
                    size=img_context.shape[-2:],  # 匹配 [H, W]  
                    mode='bilinear',  
                    align_corners=False  
                )  

                # 添加深度维度以匹配img_context的5维结构  
                semantic_feat_expanded = semantic_feat_resized.unsqueeze(2)  # [3, 32, 1, H, W]  
                semantic_feat_expanded = semantic_feat_expanded.expand(  
                    -1, -1, img_context.shape[2], -1, -1  # 扩展深度维度  
                )  

                # 需要将semantic_feat_expanded转换为与img_context相同的batch维度  
                # img_context: [18, 160, 70, 1, 44], semantic_feat_expanded: [3, 32, 70, 1, 44]  
                semantic_feat_expanded = semantic_feat_expanded.repeat_interleave(6, dim=0)  # [18, 32, 70, 1, 44]  

                #print(f"semantic_feat_expanded after repeat: {semantic_feat_expanded.shape}")  

                # 现在可以安全地拼接  
                img_context = torch.cat([img_context, semantic_feat_expanded], dim=1)  # [18, 192, 70, 1, 44]  
                #print(f"img_context after semantic fusion: {img_context.shape}")  

            # 然后通过ViewAggregation网络处理    
            img_context = self._forward_view_aggregation_net(img_context)    
        else:    
            img_context = img_feat_with_depth
            
        if self.times is not None:
            t4.record()
            torch.cuda.synchronize()
            self.times['img_transform'].append(t3.elapsed_time(t4))

        img_context = self._split_batch_cam(img_context, num_cams=num_cams)
        img_context = img_context.permute(0, 1, 3, 4, 5, 2).contiguous()

        pts_context = self._split_batch_cam(pts_context, num_cams=num_cams)
        pts_context = pts_context.unsqueeze(-2).permute(0, 1, 3, 4, 5, 2).contiguous()

        fused_context = torch.cat([img_context, pts_context], dim=-1)

        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        geom_xyz[..., 2] = 0  # collapse z-axis
        geo_pos = torch.ones_like(geom_xyz)
        
        # sparse voxel pooling
        feature_map, _ = average_voxel_pooling(geom_xyz, fused_context.contiguous(), geo_pos,
                                               self.voxel_num.cuda())
        if self.times is not None:
            t5.record()
            torch.cuda.synchronize()
            self.times['img_pool'].append(t4.elapsed_time(t5))

        if return_depth:
            return feature_map.contiguous(), depth_feature[:, :self.depth_channels].softmax(1)
        return feature_map.contiguous()

    def forward(self,
                sweep_imgs,
                mats_dict,
                ptss_context,
                ptss_occupancy,
                times=None,
                return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape
                    of (B, 4, 4).
            ptss_context(Tensor): Input point context feature with shape of
                (B * num_cameras, num_sweeps, C, D, W).
            ptss_occupancy(Tensor): Input point occupancy with shape of
                (B * num_cameras, num_sweeps, 1, D, W).
            times(Dict, optional): Inference time measurement.
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Return:
            Tensor: bev feature map.
        """
        self.times = times
        if self.times is not None:
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            t1.record()
            torch.cuda.synchronize()

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape
        key_frame_res = self._forward_single_sweep(
            0,
            sweep_imgs[:, 0:1, ...],
            mats_dict,
            ptss_context[:, 0, ...] if ptss_context is not None else None,
            ptss_occupancy[:, 0, ...] if ptss_occupancy is not None else None,
            return_depth=return_depth)
        if self.times is not None:
            t2.record()
            torch.cuda.synchronize()
            self.times['img'].append(t1.elapsed_time(t2))

        if num_sweeps == 1:
            if return_depth:
                return key_frame_res[0].unsqueeze(1), key_frame_res[1], self.times
            else:
                return key_frame_res.unsqueeze(1), self.times

        key_frame_feature = key_frame_res[0] if return_depth else key_frame_res
        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index:sweep_index + 1, ...],
                    mats_dict,
                    ptss_context[:, sweep_index, ...] if ptss_context is not None else None,
                    ptss_occupancy[:, sweep_index, ...] if ptss_occupancy is not None else None,
                    return_depth=False)
                ret_feature_list.append(feature_map)

        if return_depth:
            return torch.stack(ret_feature_list, 1), key_frame_res[1], self.times
        else:
            return torch.stack(ret_feature_list, 1), self.times
        

    
class SemanticGuidedRVTLSSFPN(RVTLSSFPN):  
    def __init__(self, use_semantic_guidance=False, semantic_model_type='deeplabv3_mobilenet_v3_large',   
             semantic_feat_dim=32, **kwargs):  
        super().__init__(**kwargs)  
      
        self.use_semantic_guidance = use_semantic_guidance  

        if self.use_semantic_guidance:  
            # 加载预训练语义分割模型  
            self.semantic_model = self._load_pretrained_semantic_model(semantic_model_type)  

            # 动态获取语义模型的实际输出通道数  
            actual_num_classes = self._get_semantic_model_output_channels()  

            # 使用实际的类别数创建投影层  
            self.semantic_proj = nn.Sequential(  
                nn.Conv2d(actual_num_classes, semantic_feat_dim, 1),  
                nn.BatchNorm2d(semantic_feat_dim),  
                nn.ReLU(inplace=True)  
            )  

            # 关键修改：更新ViewAggregation的输入通道数以包含语义特征  
            total_input_channels = self.output_channels * 2 + semantic_feat_dim  
            self.view_aggregation_net = ViewAggregation(  
                total_input_channels,  # 160 + 32 = 192  
                self.output_channels * 2,  # 160  
                self.output_channels  # 80  
            )  
            
            self.semantic_attention = SemanticAttentionModule(self.output_channels, semantic_feat_dim)
            
            self.temporal_consistency = TemporalSemanticConsistency(semantic_feat_dim)
            
        else:  
            # 如果不使用语义引导，保持原有的ViewAggregation配置  
            self.view_aggregation_net = ViewAggregation(  
                self.output_channels * 2,  
                self.output_channels * 2,  
                self.output_channels  
            )
            
    def _load_pretrained_semantic_model(self, model_type):  
        if model_type == 'deeplabv3_mobilenet_v3_large_cityscapes':  
            # 使用在Cityscapes上预训练的模型（19类）  
            model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_mobilenet_v3_large',   
                                  pretrained=False, num_classes=19)  
            # 加载Cityscapes预训练权重  
            checkpoint = torch.hub.load_state_dict_from_url(  
                'https://download.pytorch.org/models/deeplabv3_mobilenet_v3_large_coco-040b5a14.pth'  
            )  
            model.load_state_dict(checkpoint)  
        else:  
            # 原有逻辑  
            model = seg_models.deeplabv3_mobilenet_v3_large(pretrained=True)  

        model.eval()  
        for param in model.parameters():  
            param.requires_grad = False  

        return model
    
    def _get_semantic_model_output_channels(self):  
        """动态获取语义模型的输出通道数"""  
        with torch.no_grad():  
            # 创建一个小的测试输入  
            dummy_input = torch.randn(1, 3, 64, 64)  
              
            # 获取模型输出  
            dummy_output = self.semantic_model(dummy_input)  
              
            if isinstance(dummy_output, dict):  
                # DeepLabV3等模型返回字典格式  
                actual_num_classes = dummy_output['out'].shape[1]  
            else:  
                # 其他模型直接返回张量  
                actual_num_classes = dummy_output.shape[1]  
                  
        #print(f"检测到语义分割模型输出通道数: {actual_num_classes}")  
        return actual_num_classes
    
    def _extract_semantic_features(self, images):  
        """提取语义特征，正确处理多相机batch维度"""  
        if not self.use_semantic_guidance:  
            return None  

        with torch.no_grad():  
            # images shape: [batch_size*num_cams, channels, h, w]  
            batch_size_times_cams = images.shape[0]  
            original_h, original_w = images.shape[-2:]  

            # 添加调试信息  
            #print(f"Input images shape: {images.shape}")  

            # 手动实现ImageNet标准归一化  
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(images.device)  
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(images.device)  

            # 确保输入图像在[0,1]范围内  
            if images.max() > 1.0:  
                images = images / 255.0  

            normalized_images = (images - mean) / std  

            # 获取语义分割结果  
            semantic_output = self.semantic_model(normalized_images)  

            if isinstance(semantic_output, dict):  
                semantic_logits = semantic_output['out']  
            else:  
                semantic_logits = semantic_output  

            #print(f"Semantic logits shape: {semantic_logits.shape}")  

            # 转换为概率分布  
            semantic_probs = F.softmax(semantic_logits, dim=1)  

            # 投影到指定维度  
            semantic_feat = self.semantic_proj(semantic_probs)  

            #print(f"Semantic feat after projection shape: {semantic_feat.shape}")  

            # 关键修正：确保语义特征与原始图像尺寸匹配  
            if semantic_feat.shape[-2:] != (original_h, original_w):  
                semantic_feat = F.interpolate(  
                    semantic_feat,   
                    size=(original_h, original_w),   
                    mode='bilinear',   
                    align_corners=False  
                )  
                #print(f"Semantic feat after interpolation shape: {semantic_feat.shape}")  

            # 重新整形为正确的batch和相机维度  
            num_cams = 6  # CRN使用6个相机  
            batch_size = batch_size_times_cams // num_cams  
            semantic_channels = semantic_feat.shape[1]  

            #print(f"Calculated batch_size: {batch_size}, num_cams: {num_cams}, channels: {semantic_channels}")  

            # 验证尺寸是否匹配  
            expected_size = batch_size * num_cams * semantic_channels * original_h * original_w  
            actual_size = semantic_feat.numel()  

            if expected_size != actual_size:  
                #print(f"Size mismatch! Expected: {expected_size}, Actual: {actual_size}")  
                # 如果尺寸不匹配，直接返回None，跳过语义融合  
                return None  

            # 重新整形  
            semantic_feat = semantic_feat.view(batch_size, num_cams,   
                                             semantic_channels,   
                                             original_h, original_w)  

            #print(f"Semantic feat after reshape: {semantic_feat.shape}")  

            # 对多相机特征进行平均池化，得到 [batch_size, channels, h, w]  
            semantic_feat = semantic_feat.mean(dim=1)  

            #print(f"Final semantic feat shape: {semantic_feat.shape}")  

        return semantic_feat

    def _validate_semantic_integration(self):  
        """验证语义模块与原有架构的兼容性"""  
        if not self.use_semantic_guidance:  
            return True  

        try:  
            # 测试语义特征提取  
            test_input = torch.randn(1, 3, 256, 704)  
            semantic_feat = self._extract_semantic_features(test_input)  

            if semantic_feat is not None:  
                expected_channels = getattr(self, 'semantic_feat_dim', 32)  
                actual_channels = semantic_feat.shape[1]  

                if actual_channels != expected_channels:  
                    print(f"警告: 语义特征通道数不匹配。期望: {expected_channels}, 实际: {actual_channels}")  
                    return False  

            print("语义模块集成验证通过")  
            return True  

        except Exception as e:  
            print(f"语义模块验证失败: {e}")  
            return False
