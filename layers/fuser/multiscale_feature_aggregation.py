import torch  
from torch import nn  
import torch.nn.functional as F  
from mmcv.runner.base_module import ModuleList  
from mmcv.cnn import build_norm_layer  
from mmcv.cnn.bricks.transformer import build_feedforward_network, build_positional_encoding  
from mmcv.runner import auto_fp16  
from .multimodal_feature_aggregation import MFAFuser  
from ..modules.multimodal_deformable_cross_attention import DeformableCrossAttention  
  
class MultiScaleMFAFuser(nn.Module):  
    def __init__(self, num_sweeps=4, img_dims=80, pts_dims=128, embed_dims=256,  
                 num_layers=6, num_heads=4, bev_shape=(128, 128), scales=[1.0, 0.5, 0.25]):  
        super().__init__()  
          
        self.scales = scales  
        self.num_scales = len(scales)  
        self.embed_dims = embed_dims  
          
        # 为每个尺度创建独立的融合器  
        self.scale_fusers = nn.ModuleList([  
            MFAFuser(num_sweeps, img_dims, pts_dims, embed_dims,   
                    num_layers, num_heads,   
                    (int(bev_shape[0] * scale), int(bev_shape[1] * scale)))  
            for scale in scales  
        ])  
          
        # 自适应特征选择网络  
        self.feature_selector = nn.Sequential(  
            nn.Conv2d(embed_dims * self.num_scales, embed_dims, 1),  
            nn.BatchNorm2d(embed_dims),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(embed_dims, self.num_scales, 1),  
            nn.Softmax(dim=1)  
        )  
          
        # 特征融合网络  
        self.feature_fusion = nn.Sequential(  
            nn.Conv2d(embed_dims * self.num_scales, embed_dims, 3, padding=1),  
            nn.BatchNorm2d(embed_dims),  
            nn.ReLU(inplace=True)  
        )  
          
    def forward(self, feats, times=None):  
        batch_size = feats.shape[0]  
        target_h, target_w = int(128), int(128)  # 目标BEV尺寸  
          
        # 多尺度特征提取  
        scale_features = []  
        for i, (scale, fuser) in enumerate(zip(self.scales, self.scale_fusers)):  
            # 调整输入特征尺寸  
            if scale != 1.0:  
                scaled_feats = F.interpolate(  
                    feats.view(-1, feats.shape[-1], feats.shape[-2], feats.shape[-1]),  
                    scale_factor=scale, mode='bilinear', align_corners=False  
                ).view(feats.shape[0], feats.shape[1], -1, int(feats.shape[-2] * scale), int(feats.shape[-1] * scale))  
            else:  
                scaled_feats = feats  
                  
            # 通过对应尺度的融合器  
            scale_feat, _ = fuser(scaled_feats, times)  
              
            # 调整回目标尺寸  
            if scale_feat.shape[-2:] != (target_h, target_w):  
                scale_feat = F.interpolate(scale_feat, size=(target_h, target_w),   
                                         mode='bilinear', align_corners=False)  
              
            scale_features.append(scale_feat)  
          
        # 拼接多尺度特征  
        multi_scale_feat = torch.cat(scale_features, dim=1)  
          
        # 自适应特征选择  
        attention_weights = self.feature_selector(multi_scale_feat)  
          
        # 加权融合  
        weighted_features = []  
        for i, feat in enumerate(scale_features):  
            weight = attention_weights[:, i:i+1, :, :]  
            weighted_features.append(feat * weight)  
          
        # 最终融合  
        final_feat = self.feature_fusion(torch.cat(weighted_features, dim=1))  
          
        return final_feat, times