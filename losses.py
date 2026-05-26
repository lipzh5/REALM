import torch
import torch.nn.functional as F

def masked_l1_loss(pred, target, lengths):
    # pred and target are (Batch, Time, Dims)
    # lengths is (Batch,)
    mask = torch.arange(pred.size(1), device=pred.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float() # (B, T, 1)
    
    # Calculate unreduced L1 loss
    raw_loss = F.l1_loss(pred, target, reduction='none')
    
    # Apply mask and compute the mean only over valid frames
    valid_loss = (raw_loss * mask).sum() / mask.sum()
    return valid_loss


def masked_vel_loss_rot(config, pred, target, lengths):
    """
    Calculates the masked L1/MSE loss on the first derivative (velocity) of the rotation.
    """
    device = pred.device
    T = pred.shape[1]
    weights = config.train.loss_weights
    rot_dim = 3

    # 1. Create standard mask
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()  # (B, T, 1)

    # 2. Extract Rotation (64:67)
    pred_rot = pred[:, :, -6:-3]
    target_rot = target[:, :, -6:-3]

    # 3. Calculate Velocity (Differences)
    pred_vel = torch.diff(pred_rot, dim=1)
    target_vel = torch.diff(target_rot, dim=1)

    # 4. Shift Mask for Velocity
    # Since diff drops the length to T-1, we drop the first frame of the mask
    mask_vel = mask[:, 1:, :] 
    valid_vel_frames = mask_vel.sum().clamp(min=1)

    # 5. Apply Axis Weighting
    w_rot_axis = torch.tensor(weights.rot_axis, device=device).view(1, 1, 3)

    # 6. Calculate Loss
    mse_vel_raw = (pred_vel - target_vel) ** 2
    loss_vel = (mse_vel_raw * w_rot_axis * mask_vel).sum() / (valid_vel_frames * rot_dim)

    return loss_vel

def masked_mse_loss(config, pred, target, lengths):
    """
    pred, target: (B, T, D) where D=70
    lengths: (B,)
    w_rot_axis: list or tensor of size 3, e.g., [2.0, 2.0, 1.0] for [Pitch, Yaw, Roll]
    """
    weights = config.train.loss_weights
    B, T, D = pred.shape
    device = pred.device

    # Create mask based on sequence lengths
    mask = torch.arange(T, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    mask = mask.unsqueeze(-1).float()  # (B, T, 1)

    valid_frames = mask.sum().clamp(min=1)

    # --- expression (0:64) ---
    exp_dim = 64
    mse_exp = (pred[:, :, :64] - target[:, :, :64]) ** 2
    loss_exp = (mse_exp * mask).sum() / (valid_frames * exp_dim)

    # --- rotation (64:67) ---
    rot_dim = 3
    pred_rot = pred[:, :, 64:67]
    target_rot = target[:, :, 64:67]
    
    mse_rot_raw = (pred_rot - target_rot) ** 2 # (B, T, 3)
    
    # 1. Axis-Specific Weighting
    # Boost Pitch (0) and Yaw (1) which are usually for Nod/Shake
    w_rot_axis = torch.tensor(weights.rot_axis, device=device)
    w_rot_axis = w_rot_axis.view(1, 1, 3)

    loss_rot = (mse_rot_raw * w_rot_axis * mask).sum() / (valid_frames * rot_dim)

    # --- translation (67:70) ---
    tran_dim = 3
    mse_tran = (pred[:, :, 67:70] - target[:, :, 67:70]) ** 2
    loss_tran = (mse_tran * mask).sum() / (valid_frames * tran_dim)

    # Final Weighted Sum
    
    w_rot = weights.rot
    loss = (
        weights.exp * loss_exp +
        w_rot * loss_rot +
        weights.tran * loss_tran 
    )
    # print(f'loss: {loss}, exp: {weights.exp * loss_exp}, rot: {weights.rot * loss_rot}, tran: {weights.tran * loss_tran}')
    return loss, loss_exp.detach(), loss_rot.detach(), loss_tran.detach()