
import torch
import torch.nn as nn
import math
import numpy as np
import os
import os.path as osp
import torch.nn.functional as F
import random
from models.modules.refine_module import MotionRefineNet
from models.modules.speaker_encoder import SpeakerEncoder
from models.modules.speaker_listener_fusion import SpeakerListenerFusion
from models.modules.listener_decoder import ReactionDecoder

cur_dir = osp.dirname(osp.abspath(__file__))



class ListenerMotionGenerator(nn.Module):
    """Online, listener motion generation"""
    def __init__(self, config, device): 
        super().__init__()
        
        self.window_size = config.model.window_size 
        self.pred_delta_motion = config.model.pred_delta_motion
        self.exp_dim = config.model.exp_dim
        self.device = device
        self.pred_action_token = getattr(config.model, 'pred_action_token', False)
        
        self.speaker_encoder = SpeakerEncoder(motion_dim=config.model.motion_dim, audio_dim=config.model.audio_dim, feature_dim=config.model.feature_dim)
        self.speaker_listener_fusion = SpeakerListenerFusion(config.model.motion_dim, config.model.feature_dim, delay_frames=getattr(config.model, 'fusion_delay_frames', 8))
        self.listener_future_motion_decoder = ReactionDecoder(config, out_motion_dim=config.model.motion_dim, window_size=config.model.window_size, device=device)
        
        self.overlap_window = getattr(config.model, 'overlap_window', 0)
        self.stride = self.window_size - self.overlap_window 
        self.history_len = config.model.max_listener_window
    
        self.exp_dim = getattr(config.model, 'exp_dim', 64)
        
        self.waypoint_predictor = nn.Sequential(
            nn.Linear(config.model.audio_dim + 3, 128), 
            nn.GELU(),
            nn.Linear(128, 3)
        )
       
        self.lambda_bd_rot = getattr(config.train.loss_weights, 'boundary_rot', 1.0)
        self.lambda_bd_exp = getattr(config.train.loss_weights, 'boundary_exp', 0.5)
        self.lambda_bd_trans = getattr(config.train.loss_weights, 'boundary_trans', 0.1)
       
        self.refine_net = MotionRefineNet(config.model.refine_dim, audio_dim=config.model.audio_dim)

    def boostrap_init(self, frame_zero, bootstrap_audio):
        B, _, D = frame_zero.shape
        W = self.window_size
        
        # 1. Isolate the exact parameter groups
        expr_0 = frame_zero[:, 0, :self.exp_dim]                 # (B, 64)
        rot_0 = frame_zero[:, 0, self.exp_dim:self.exp_dim+3]    # (B, 3) - ONLY Rotation
        tran_0 = frame_zero[:, 0, self.exp_dim+3:self.exp_dim+6] # (B, 3) - Translation
        
        # 2. Predict the 3D Rotation Waypoint
        global_audio_context = bootstrap_audio.mean(dim=1) 

         # NOTE [ablation] 2: deaf listener, no audio context
        # waypoint_input = rot_0      
        waypoint_input = torch.cat([rot_0, global_audio_context], dim=-1)

        pred_rot_w = self.waypoint_predictor(waypoint_input)    # (B, 3)
        # 3. Create the Rotation Interpolation (The Guide Curve)
        steps = torch.linspace(0, 1, steps=W, device=self.device).view(1, W, 1)
        rot_guide = (1.0 - steps) * rot_0.unsqueeze(1) + steps * pred_rot_w.unsqueeze(1) 

        # 4. Create the Flatlines (Expression and Translation wait for AR residuals)
        expr_guide = expr_0.unsqueeze(1).repeat(1, W, 1)         
        tran_guide = tran_0.unsqueeze(1).repeat(1, W, 1)         
        
        # 5. Concatenate to form the full Guide Curve
        if D > self.exp_dim + 6:
            crop_0 = frame_zero[:, 0, self.exp_dim+6:]
            crop_guide = crop_0.unsqueeze(1).repeat(1, W, 1)
            guide_curve = torch.cat([expr_guide, rot_guide, tran_guide, crop_guide], dim=-1)
        else:
            guide_curve = torch.cat([expr_guide, rot_guide, tran_guide], dim=-1)
        
        # ==========================================
        # FEED INTO AR DECODER
        # ==========================================
        speaker_behaviour_feat_boot = self.speaker_encoder(bootstrap_audio)
        speaker_listener_feat_boot = self.speaker_listener_fusion(speaker_behaviour_feat_boot, guide_curve)
        
        # The decoder now predicts the high-frequency residuals on top of this guide
        pred_expr_b, pred_rot_b, pred_tran_b, gate_alpha = self.listener_future_motion_decoder(
            speaker_listener_feat_boot, guide_curve
        )

        # pred delta motion
        expr_dim = pred_expr_b.shape[-1]
        rot_dim = pred_rot_b.shape[-1]
        tran_dim = pred_tran_b.shape[-1]
        
        expr_end = expr_dim
        rot_end = expr_end + rot_dim
        tran_end = rot_end + tran_dim
        
        past_expr_b = frame_zero[:, -1:, :expr_end]
        past_rot_b = frame_zero[:, -1:, expr_end:rot_end]
        past_tran_b = frame_zero[:, -1:, rot_end:tran_end]
        
        pred_expr_abs_b = past_expr_b + torch.cumsum(pred_expr_b, dim=1)
        pred_rot_abs_b = past_rot_b + torch.cumsum(pred_rot_b, dim=1)
        pred_tran_abs_b = past_tran_b + torch.cumsum(pred_tran_b, dim=1)
        
        first_window_motion = torch.cat((pred_expr_abs_b, pred_rot_abs_b, pred_tran_abs_b), dim=-1)
        # CRITICAL: Hard-anchor the 0-th frame to exact ground truth so the trajectory starts perfectly
        first_window_motion[:, 0, :] = frame_zero[:, 0, :]
        
        return first_window_motion, pred_rot_w
       

    def forward(self, speaker_audio, listener_motion_gt, lengths, detach_refiner=False):
        """
        Args
            speaker_audio: (B, T*4, 768)
            listener_motion_gt: (B, T, 70)
            lengths: (B,)  <- True sequence lengths without padding
        """
        _, T, _ = listener_motion_gt.shape
        W = self.window_size
        bootstrap_history = listener_motion_gt[:, 0:1, :]
        
        # FIX: Multiply W by audio_ratio to extract the correct temporal footprint
        bootstrap_audio = speaker_audio[:, :W, :]
        listener_react_motion, pred_rot_w = self.boostrap_init(bootstrap_history, bootstrap_audio)
       
        K = self.history_len 
        gate_alpha = None
        
        boundary_loss = torch.tensor(0.0, device=speaker_audio.device)
        valid_boundary_steps = 0 # Keep track of how many steps actually had valid boundaries

        target_len = lengths.max().int().item()

        # Force the loop to run enough times to cover the remainder
        if target_len > W:
            episodes = math.ceil((target_len - W) / self.stride) + 1
        else:
            episodes = 1
        
        for i in range(1, episodes):
            curr_end_frame = W + i * self.stride  
            context_length = min(K * W, curr_end_frame)
            curr_start_frame = curr_end_frame - context_length # (0, 60) (60, 120)

            # padding
            speaker_audio_ = speaker_audio[:, curr_start_frame : curr_end_frame]
            expected_audio_len = curr_end_frame - curr_start_frame 
            # If we overshot the actual audio length, pad the temporal dimension with zeros
            if speaker_audio_.shape[1] < expected_audio_len:
                pad_size = expected_audio_len - speaker_audio_.shape[1]
                # Shape-agnostic padding (works whether audio is [B, T] or [B, T, C])
                padding = torch.zeros(
                    speaker_audio_.shape[0], 
                    pad_size, 
                    *speaker_audio_.shape[2:], 
                    device=speaker_audio_.device,
                    dtype=speaker_audio_.dtype
                )
                speaker_audio_ = torch.cat([speaker_audio_, padding], dim=1)
            
            speaker_behaviour_feat = self.speaker_encoder(speaker_audio_) #  (15, 30)
            
            past_listener_reaction = listener_react_motion
            aligned_past_listener = past_listener_reaction[:, -context_length:]
            listener_T = aligned_past_listener.shape[1]
            speaker_listener_feat = self.speaker_listener_fusion(
                speaker_behaviour_feat[:, -listener_T:],
                aligned_past_listener 
            )
            
            pred_expr, pred_rot, pred_tran, gate_alpha = self.listener_future_motion_decoder(
                speaker_listener_feat[:, -listener_T:],
                aligned_past_listener 
            )
            
            anchor_idx = -self.overlap_window - 1 if self.overlap_window > 0 else -1
            expr_dim = pred_expr.shape[-1]
            rot_dim = pred_rot.shape[-1]
            tran_dim = pred_tran.shape[-1]
            
            # Slicing indices
            expr_end = expr_dim
            rot_end = expr_end + rot_dim
            tran_end = rot_end + tran_dim
            
            # Extract the anchor frames from history
            past_expr = past_listener_reaction[:, [anchor_idx], :expr_end]
            past_rot = past_listener_reaction[:, [anchor_idx], expr_end:rot_end]
            past_tran = past_listener_reaction[:, [anchor_idx], rot_end:tran_end]
            
            # Integrate the network's delta predictions to get absolute positions
            pred_expr_abs = past_expr + torch.cumsum(pred_expr, dim=1)
            pred_rot_abs = past_rot + torch.cumsum(pred_rot, dim=1)
            pred_tran_abs = past_tran + torch.cumsum(pred_tran, dim=1)
            
            # Concatenate the newly integrated absolute poses (assuming crop remains absolute)
            pred_motion = torch.cat((pred_expr_abs, pred_rot_abs, pred_tran_abs), dim=-1)
        

            # --- NEW: Masked Boundary Loss ---
            if self.training and past_listener_reaction.shape[1] >= 2 and pred_motion.shape[1] >= 2:
                # 2. Determine the global frame index of this specific boundary
                # The boundary is exactly where the new predicted window starts
                boundary_idx = W + (i - 1) * self.stride 
                
                # 3. Create a mask: Is this boundary frame index within the valid length? (B,)
                valid_mask = (boundary_idx < lengths).float()
                
                if valid_mask.sum() > 0: # Only calculate if at least one item in batch is valid
                    w_rot = self.lambda_bd_rot
                    w_expr = self.lambda_bd_exp
                    w_tran = self.lambda_bd_trans # Keep this low due to dataset translation noise
                    
                    exp_dim = self.exp_dim
                    
                    # Helper function to compute separated C0 / C1 losses
                    def compute_split_loss(pred, past):
                        # Calculate unreduced MSE
                        raw_loss = F.mse_loss(pred, past, reduction='none')
                        
                        # Split and mean across their specific dimensions
                        l_expr = raw_loss[:, :exp_dim].mean(dim=-1)
                        l_rot = raw_loss[:, exp_dim:exp_dim+3].mean(dim=-1)
                        l_tran = raw_loss[:, exp_dim+3:exp_dim+6].mean(dim=-1)
        
                        return (w_expr * l_expr) + (w_rot * l_rot) + (w_tran * l_tran)

                    # 2. Compute C0 (Position)
                    loss_c0_batch = compute_split_loss(
                        pred_motion[:, 0, :], 
                        past_listener_reaction[:, -1, :]
                    )
                    
                    # 3. Compute C1 (Velocity)
                    past_vel = past_listener_reaction[:, -1, :] - past_listener_reaction[:, -2, :]
                    pred_vel = pred_motion[:, 1, :] - pred_motion[:, 0, :]
                    loss_c1_batch = compute_split_loss(pred_vel, past_vel)
                    
                    # 4. Apply mask and average over valid batch items
                    step_loss = ((loss_c0_batch + loss_c1_batch) * valid_mask).sum() / valid_mask.sum()
                    boundary_loss += step_loss
                    valid_boundary_steps += 1
            # Strict Concatenation
            new_motion = pred_motion[:, -self.stride:, :]
            listener_react_motion = torch.cat([past_listener_reaction, new_motion], dim=1)
        
        # 4. Average over the valid steps (instead of total episodes)
        if valid_boundary_steps > 0:
            boundary_loss = boundary_loss / valid_boundary_steps
        else:
            # Fallback to 0 loss. 
            boundary_loss = (listener_react_motion.sum() * 0.0)
        
        final_listener_motion = listener_react_motion.clone()
        
        if final_listener_motion.shape[1] > target_len:
            final_listener_motion = final_listener_motion[:, :target_len, :]

        return final_listener_motion, gate_alpha, boundary_loss, pred_rot_w
     
     
