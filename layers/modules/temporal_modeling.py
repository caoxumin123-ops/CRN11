import torch  
import torch.nn as nn  
import torch.nn.functional as F  
  
class MotionPredictionModule(nn.Module):    
    def __init__(self, feature_dim=256, hidden_dim=128, num_frames=4):    
        super().__init__()    
        self.feature_dim = feature_dim    
        self.hidden_dim = hidden_dim    
        self.num_frames = num_frames  # 存储帧数参数  
            
        # 光流估计网络    
        self.flow_estimator = nn.Sequential(    
            nn.Conv2d(feature_dim * 2, hidden_dim, 3, padding=1),    
            nn.BatchNorm2d(hidden_dim),    
            nn.ReLU(inplace=True),    
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),    
            nn.BatchNorm2d(hidden_dim),    
            nn.ReLU(inplace=True),    
            nn.Conv2d(hidden_dim, 2, 3, padding=1)  # 输出2通道光流    
        )    
            
        # 运动预测网络    
        self.motion_predictor = nn.Sequential(    
            nn.Conv2d(feature_dim + 2, hidden_dim, 3, padding=1),    
            nn.BatchNorm2d(hidden_dim),    
            nn.ReLU(inplace=True),    
            nn.Conv2d(hidden_dim, feature_dim, 3, padding=1)    
        )  
          
        # 时序特征融合网络 - 利用num_frames参数  
        self.temporal_fusion = nn.Sequential(  
            nn.Conv2d(feature_dim * num_frames, hidden_dim, 1),  
            nn.BatchNorm2d(hidden_dim),  
            nn.ReLU(inplace=True),  
            nn.Conv2d(hidden_dim, feature_dim, 1)  
        )  
            
    def forward(self, current_feat, prev_feat=None, frame_sequence=None):    
        if frame_sequence is not None:  
            # 处理多帧序列的情况  
            return self.forward_sequence(frame_sequence)  
        elif prev_feat is not None:  
            # 处理两帧的情况  
            return self.forward_pair(current_feat, prev_feat)  
        else:  
            # 只有当前帧的情况  
            return current_feat, None  
      
    def forward_pair(self, current_feat, prev_feat):  
        # 估计光流    
        flow_input = torch.cat([current_feat, prev_feat], dim=1)    
        optical_flow = self.flow_estimator(flow_input)    
            
        # 基于光流预测运动    
        motion_input = torch.cat([current_feat, optical_flow], dim=1)    
        predicted_motion = self.motion_predictor(motion_input)    
            
        return predicted_motion, optical_flow  
      
    def forward_sequence(self, frame_sequence):  
        # frame_sequence: [B, T, C, H, W] where T = num_frames  
        B, T, C, H, W = frame_sequence.shape  
          
        # 确保帧数匹配  
        if T != self.num_frames:  
            # 如果帧数不匹配，进行插值或截取  
            if T > self.num_frames:  
                # 均匀采样  
                indices = torch.linspace(0, T-1, self.num_frames).long()  
                frame_sequence = frame_sequence[:, indices]  
            else:  
                # 重复最后一帧  
                last_frame = frame_sequence[:, -1:].repeat(1, self.num_frames - T, 1, 1, 1)  
                frame_sequence = torch.cat([frame_sequence, last_frame], dim=1)  
          
        # 将所有帧拼接进行融合  
        fused_input = frame_sequence.view(B, -1, H, W)  # [B, T*C, H, W]  
        fused_features = self.temporal_fusion(fused_input)  
          
        # 计算相邻帧之间的光流  
        optical_flows = []  
        for t in range(1, self.num_frames):  
            flow_input = torch.cat([frame_sequence[:, t], frame_sequence[:, t-1]], dim=1)  
            flow = self.flow_estimator(flow_input)  
            optical_flows.append(flow)  
          
        return fused_features, optical_flows
  
class TemporalConsistencyLoss(nn.Module):  
    def __init__(self, weight=1.0):  
        super().__init__()  
        self.weight = weight  
          
    def forward(self, current_pred, prev_pred, optical_flow):  
        # 使用光流warp前一帧预测  
        warped_prev = self.warp_features(prev_pred, optical_flow)  
          
        # 计算时序一致性损失  
        consistency_loss = F.mse_loss(current_pred, warped_prev)  
          
        return self.weight * consistency_loss  
      
    def warp_features(self, features, flow):  
        B, C, H, W = features.shape  
          
        # 创建网格  
        grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))  
        grid = torch.stack([grid_x, grid_y], dim=0).float()  
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1).to(features.device)  
          
        # 应用光流  
        new_grid = grid + flow  
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0  
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0  
        new_grid = new_grid.permute(0, 2, 3, 1)  
          
        # 使用grid_sample进行warp  
        warped = F.grid_sample(features, new_grid, align_corners=True)  
        
        return warped  
  
class AdaptiveFrameSelector(nn.Module):    
    def __init__(self, feature_dim=256, num_frames=4):    
        super().__init__()    
        self.num_frames = num_frames    
        self.feature_dim = feature_dim  # 存储特征维度  
            
        # 信息量评估网络    
        self.info_evaluator = nn.Sequential(    
            nn.AdaptiveAvgPool2d(1),    
            nn.Conv2d(feature_dim, feature_dim // 4, 1),    
            nn.ReLU(inplace=True),    
            nn.Conv2d(feature_dim // 4, 1, 1),    
            nn.Sigmoid()    
        )  
          
        # 帧间相关性评估  
        self.correlation_evaluator = nn.Sequential(  
            nn.Conv2d(feature_dim * 2, feature_dim // 2, 1),  
            nn.ReLU(inplace=True),  
            nn.AdaptiveAvgPool2d(1),  
            nn.Conv2d(feature_dim // 2, 1, 1),  
            nn.Sigmoid()  
        )  
            
    def forward(self, frame_features):    
        # frame_features: [B, T, C, H, W]    
        B, T, C, H, W = frame_features.shape    
          
        # 确保特征维度匹配  
        assert C == self.feature_dim, f"Expected {self.feature_dim} channels, got {C}"  
            
        # 计算每帧的信息量分数    
        info_scores = []    
        for t in range(T):    
            score = self.info_evaluator(frame_features[:, t])    
            info_scores.append(score)    
            
        info_scores = torch.cat(info_scores, dim=1)  # [B, T, 1, 1]  
          
        # 计算帧间相关性（避免选择过于相似的帧）  
        correlation_penalties = torch.zeros_like(info_scores)  
        for t in range(T):  
            for s in range(t+1, T):  
                corr_input = torch.cat([frame_features[:, t], frame_features[:, s]], dim=1)  
                corr_score = self.correlation_evaluator(corr_input)  
                correlation_penalties[:, t] += corr_score  
                correlation_penalties[:, s] += corr_score  
          
        # 综合信息量和相关性得分  
        final_scores = info_scores - 0.1 * correlation_penalties  
            
        # 选择信息量最高的帧    
        selected_indices = torch.topk(final_scores.squeeze(-1).squeeze(-1),   
                                    k=min(self.num_frames, T), dim=1)[1]    
            
        # 提取选中的帧    
        selected_frames = []    
        for b in range(B):    
            selected = frame_features[b, selected_indices[b]]    
            selected_frames.append(selected)    
            
        return torch.stack(selected_frames, dim=0), selected_indices

class TemporalModelingIntegrator(nn.Module):  
    """整合时序建模的各个组件"""  
    def __init__(self, feature_dim=256, hidden_dim=128, num_frames=4):  
        super().__init__()  
        self.num_frames = num_frames  
          
        self.motion_predictor = MotionPredictionModule(  
            feature_dim=feature_dim,   
            hidden_dim=hidden_dim,   
            num_frames=num_frames  
        )  
          
        self.frame_selector = AdaptiveFrameSelector(  
            feature_dim=feature_dim,   
            num_frames=num_frames  
        )  
          
        self.temporal_loss = TemporalConsistencyLoss(weight=1.0)  
      
    def forward(self, frame_sequence, training=True):  
        """  
        Args:  
            frame_sequence: [B, T, C, H, W] 输入帧序列  
            training: 是否为训练模式  
        Returns:  
            enhanced_features: 增强后的特征  
            losses: 训练时的损失字典  
        """  
        B, T, C, H, W = frame_sequence.shape  
          
        # 1. 自适应帧选择  
        if T > self.num_frames:  
            selected_frames, selected_indices = self.frame_selector(frame_sequence)  
        else:  
            selected_frames = frame_sequence  
            selected_indices = None  
          
        # 2. 运动预测和特征增强  
        enhanced_features, optical_flows = self.motion_predictor(  
            current_feat=None,   
            prev_feat=None,   
            frame_sequence=selected_frames  
        )  
          
        losses = {}  
        if training and optical_flows:  
            # 3. 计算时序一致性损失  
            temporal_loss = 0  
            for i, flow in enumerate(optical_flows):  
                if i + 1 < selected_frames.shape[1]:  
                    current_pred = selected_frames[:, i+1]  
                    prev_pred = selected_frames[:, i]  
                    temporal_loss += self.temporal_loss(current_pred, prev_pred, flow)  
              
            losses['temporal_consistency'] = temporal_loss / len(optical_flows)  
          
        return enhanced_features, losses
