import os
import os.path as osp
import time
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from utils import *
from models.realm import ListenerMotionGenerator
from models.discriminator import TemporalDiscriminator
from losses import *
from datasets.vico_dataset import get_mean_std
from eval_util import *


@torch.no_grad()
def test(config, val_loader, model, device, epoch, debug=False):
    # ==========================================
    # 1. INITIALIZE TRACKERS & COLLECTORS
    # ==========================================
    # L1 Loss Trackers
    losses_exp = AverageMeter()
    losses_rot = AverageMeter()
    losses_tran = AverageMeter()
    losses_pose = AverageMeter()

    all_pred_exp, all_gt_exp = [],[]
    # ==========================================
    # 2. MODEL & DATA PREPARATION
    # ==========================================
    model.eval()
    W = model.window_size
    overlap = getattr(config.model, 'overlap_window', 0)
    
    _root_dir = osp.join(CUR_DIR, '../data/vico')
    mean_face, std_face = get_mean_std(_root_dir, who='listener')
    mean_face = torch.from_numpy(mean_face).to(device); std_face = torch.from_numpy(std_face).to(device)
    reverse_transform_3dmm = transforms.Lambda(lambda e: (e * std_face) + mean_face)
   
    # ==========================================
    # 3. VALIDATION LOOP
    # ==========================================
    for bid, batch in enumerate(tqdm(val_loader)):
        speaker_audio_clip = batch['audio'].to(device)
        speaker_motion_clip = batch['speaker_face'].to(device)
        listener_motion_clip = batch['listener_face'].to(device) 
        lengths = batch['lengths'].to(device)
        clip_length = int(batch['lengths'][0].item())

        # Forward Pass & Reverse Transform
        pred_motion_full, _, _, _ = model(speaker_audio_clip, listener_motion_clip, lengths)
        pred_motion_full = reverse_transform_3dmm(pred_motion_full)
        listener_motion_full = reverse_transform_3dmm(listener_motion_clip)
        speaker_motion_full = reverse_transform_3dmm(speaker_motion_clip)

        # Temporal Truncation
        eval_start = 1
        eval_end = clip_length
        
        # We need at least 2 frames for temporal variance/difference calculation
        if eval_end - eval_start <= 1:
            continue 

        # Slice evaluation windows
        pred_eval = pred_motion_full[:, eval_start:eval_end, :]
        gt_eval = listener_motion_full[:, eval_start:eval_end, :]
        speaker_eval = speaker_motion_full[:, eval_start:eval_end, :]

        # Extract features
        pred_exp, gt_exp = pred_eval[:, :, :64], gt_eval[:, :, :64]
        pred_rot, gt_rot = pred_eval[:, :, 64:67], gt_eval[:, :, 64:67]
        pred_tran, gt_tran = pred_eval[:, :, 67:70], gt_eval[:, :, 67:70]
        pred_pose, gt_pose = pred_eval[:, :, 64:70], gt_eval[:, :, 64:70]
        B = pred_exp.shape[0]
        for i in range(B):
            all_pred_exp.append(pred_exp[i].cpu().numpy()); all_gt_exp.append(gt_exp[i].cpu().numpy())
     
        # --- A. Standard L1 Losses ---
        losses_exp.update(F.l1_loss(pred_exp, gt_exp).item())
        losses_rot.update(F.l1_loss(pred_rot, gt_rot).item())
        losses_tran.update(F.l1_loss(pred_tran, gt_tran).item())
        losses_pose.update(F.l1_loss(pred_pose, gt_pose).item())
  

    fid_delta_exp = calculate_fid_delta([t[0] for t in all_pred_exp], [t[0] for t in all_gt_exp])
    # Final Dictionary
    metrics = {
        'l1_exp': losses_exp.avg, 'l1_rot': losses_rot.avg, 'l1_tran': losses_tran.avg, 'l1_pose': losses_pose.avg, 'fid_delta_exp': fid_delta_exp}
    return metrics

def train_one_epoch_base(config, train_loader, model, optimizer, scheduler, device, epoch):
    W = config.model.window_size
    losses = AverageMeter()
    losses_exp = AverageMeter()
    losses_rot = AverageMeter()
    losses_tran = AverageMeter()
    alpha_tracker = AverageMeter()
    losses_vel = AverageMeter()
    losses_wp = AverageMeter()
    losses_bd = AverageMeter()
    
    model.train()

    lambda_bd = getattr(config.train.loss_weights, 'boundary', 5.0)
    lambda_wp = getattr(config.train.loss_weights, 'wp', 1.0)
 
    for bid, batch in enumerate(tqdm(train_loader, desc=f"Base Epoch {epoch}")):
        speaker_audio_clip = batch['audio'].to(device)
        listener_motion = batch['listener_face'].to(device)
        speaker_clip_length = batch['lengths'].to(device) 
        
        pred_motion, gate_alpha, boundary_loss, pred_rot_w = model(speaker_audio_clip, listener_motion, speaker_clip_length)

        exp_dim = model.exp_dim
        gt_rot_w = listener_motion[:, W-1, exp_dim:exp_dim+3] 
        waypoint_loss = F.mse_loss(pred_rot_w, gt_rot_w) 
        
        start = 1
 
        loss_pred = pred_motion[:, start:, :]
        loss_target = listener_motion[:, start:, :]
        loss_lengths = torch.clamp(speaker_clip_length - start, min=0)
        
        loss, loss_exp, loss_rot, loss_tran = masked_mse_loss(config, loss_pred, loss_target, loss_lengths)
        vel_loss = masked_vel_loss_rot(config, loss_pred, loss_target, loss_lengths)

        loss += lambda_bd * boundary_loss
        loss += lambda_wp * waypoint_loss
        loss = loss + 10 * vel_loss
        if gate_alpha is not None:
            loss = loss + 0.01 * (1 - gate_alpha.mean())
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        assert config.train.lr_policy == 'cosine', "LR policy must be [Cosine] here"
        scheduler.step()
    
        batch_size = len(speaker_audio_clip)
        losses_exp.update(loss_exp.item(), batch_size)
        losses_rot.update(loss_rot.item(), batch_size)
        losses_tran.update(loss_tran.item(), batch_size)
        losses_vel.update(vel_loss.item(), batch_size)
        losses_bd.update(boundary_loss.item(), batch_size)
        losses_wp.update(waypoint_loss.item(), batch_size)
        losses.update(loss.item(), batch_size)
        
        if gate_alpha is not None:
            alpha_tracker.update(gate_alpha.detach().mean().item())
            
    print(f'Epoch: {epoch} | Vel Loss: {losses_vel.avg:.4f} | BD: {losses_bd.avg:.4f} | WP: {losses_wp.avg:.4f}') 
    return losses.avg, losses_exp.avg, losses_rot.avg, losses_tran.avg, alpha_tracker.avg


def train_one_epoch_refine(config, train_loader, model, discriminator, optimizer_G, optimizer_D, scheduler_G, scheduler_D, device, epoch, adversarial_loss):
    W = config.model.window_size
    losses_g_total = AverageMeter()
    losses_g_l1 = AverageMeter()
    losses_g_adv = AverageMeter()
    losses_d_total = AverageMeter()

    # Base model stays in eval, only refine_net and discriminator train
    model.eval() 
    model.refine_net.train() 
    discriminator.train()

    exp_dim = model.exp_dim
    refine_dim = config.model.refine_dim
    
    for bid, batch in enumerate(tqdm(train_loader, desc=f"Refine Epoch {epoch}")):
        speaker_audio_clip = batch['audio'].to(device)
        listener_motion = batch['listener_face'].to(device)
        speaker_clip_length = batch['lengths'].to(device) 
        B = speaker_audio_clip.shape[0]
        
        with torch.no_grad():
            pred_motion, gate_alpha, boundary_loss, pred_rot_w = model(speaker_audio_clip, listener_motion, speaker_clip_length)

        refine_input = pred_motion[:, :, :exp_dim]
        loss_target_refine = listener_motion[:, :, :exp_dim]
        if refine_dim == 67: # exp + trans
            refine_input = torch.cat([refine_input, pred_motion[:, :, -3:]], dim=-1) 
            loss_target_refine = torch.cat([loss_target_refine, listener_motion[:, :, -3:]], dim=-1)

        pred_motion_refine = model.refine_net(refine_input, speaker_audio_clip)
        
        start = 1
        loss_pred_refine = pred_motion_refine[:, start:, :]
        loss_target_refine = loss_target_refine[:, start:, :]
        loss_lengths = torch.clamp(speaker_clip_length - start, min=0)

        valid_soft = torch.empty(B, 1, device=device).uniform_(0.9, 1.0)
        fake_soft = torch.empty(B, 1, device=device).uniform_(0.0, 0.1)
        valid_strict = torch.ones(B, 1, device=device)

        # Phase 1: Train Discriminator
        optimizer_D.zero_grad()
        real_pred = discriminator(loss_target_refine)
        d_real_loss = adversarial_loss(real_pred, valid_soft) 
        
        fake_pred = discriminator(loss_pred_refine.detach())
        d_fake_loss = adversarial_loss(fake_pred, fake_soft) 
        d_loss = (d_real_loss + d_fake_loss) / 2

        if d_loss.item() > 0.3:
            d_loss.backward()
            optimizer_D.step()
            if scheduler_D is not None:
                scheduler_D.step() 

        # Phase 2: Train Generator (Refine_Net)
        optimizer_G.zero_grad()
        fake_pred_for_G = discriminator(loss_pred_refine)
        g_adv_loss = adversarial_loss(fake_pred_for_G, valid_strict) 
        g_l1_loss = masked_l1_loss(loss_pred_refine, loss_target_refine, loss_lengths)
        
        lambda_adv = 0.5 
        g_total_loss = g_l1_loss + (lambda_adv * g_adv_loss)
        
        g_total_loss.backward()
        optimizer_G.step()
        if scheduler_G is not None:
            scheduler_G.step() 

        # Phase 3: Update Trackers
        losses_d_total.update(d_loss.item(), B)
        losses_g_total.update(g_total_loss.item(), B)
        losses_g_l1.update(g_l1_loss.item(), B)
        losses_g_adv.update(g_adv_loss.item(), B)

    print(f'Epoch: {epoch} | D Loss: {losses_d_total.avg:.4f} | G Total: {losses_g_total.avg:.4f} | G L1: {losses_g_l1.avg:.4f} | G Adv: {losses_g_adv.avg:.4f}') 
    return losses_g_total.avg, losses_g_l1.avg, losses_g_adv.avg, losses_d_total.avg


def get_data_loader(config, split):
    from datasets.vico_dataset import get_data_loader as get_data_loader_
    if split != 'train':
        config.train.batch_size = 1
    return get_data_loader_(config, split)


def run_test(ckpt_folder, max_epoch, device, data_split, epoch=0):
    ckpt_path = osp.join(ckpt_folder, f'realm_ep{max_epoch}.pt')
    if not osp.exists(ckpt_path):
        ckpt_path = osp.join(ckpt_folder, 'realm_best.pt')
        
    config = OmegaConf.load(osp.join(ckpt_folder, 'config.yaml'))
    model = ListenerMotionGenerator(config, device).to(device)
   
    test_loader = get_data_loader(config, split=data_split)
    base_state = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = model.state_dict()
    for k, v in state_dict.items():
        state_dict[k] = base_state.get(k, v)
    model.load_state_dict(state_dict)
    
    eval_metrics = test(config, test_loader, model, device=device, epoch=epoch)
    print('Test Results:')
    for k, val in eval_metrics.items():
        print(f'{k}: {val:.4f}')


        
def main(args):
    # === Core Setup ===
    seed = 23456
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    config = OmegaConf.load(args.config)
    
    writer = SummaryWriter(f'runs/realm/{config.data.dataset}/{args.trial}_{args.stage}')
    device = torch.device(f'cuda:{args.gpu_id}')

    if args.test:
        run_test(ckpt_folder=args.refine_ckpt_folder if args.stage == 'refine' else args.ckpt_folder, 
                 max_epoch=args.max_epoch, device=device, data_split=args.data_split, epoch=args.max_epoch)
        return

    train_loader = get_data_loader(config, split='train')
    test_loader = get_data_loader(config, split='test') 

    model = ListenerMotionGenerator(config, device).to(device)
    print(f"Model size: {get_model_size(model):.3f} MB \n *****") 
    
    ckpt_folder = osp.join(config.train.ckpt_folder, f'{config.data.dataset}_70d', f"{args.trial}_{args.stage}")
    os.makedirs(ckpt_folder, exist_ok=True)
    config.train.ckpt_folder = ckpt_folder

    shutil.copy(args.config, osp.join(ckpt_folder, 'config.yaml')) 
    shutil.copy(osp.abspath(__file__), osp.join(ckpt_folder, 'train_xgen.py'))

    # Extract specific epoch counts
    base_epochs = config.train.base_epochs
    refine_epochs = config.train.refine_epochs
    warmup_ratio = getattr(config.train, 'warmup_ratio', 0.1)
    
    ts = time.time()

    # ==========================================
    # STAGE 1: BASE MODEL TRAINING
    # ==========================================
    # if args.stage in ['base', 'both']:
    print(f"--- Starting Base Model Training ({base_epochs} Epochs) ---")
    best_base_loss = float('inf')
    total_base_steps = len(train_loader) * base_epochs
    warmup_base_steps = int(total_base_steps * warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.train.lr, weight_decay=1e-4)
    scheduler = get_lr_scheduler(optimizer, config.train.lr_policy, warmup_steps=warmup_base_steps, total_steps=total_base_steps, **config.train.step_scheduler_params)

    for epoch in range(base_epochs):
        losses, loss_exp, loss_rot, loss_tran, alpha = train_one_epoch_base(
            config, train_loader, model, optimizer, scheduler, device, epoch
        )
        
        writer.add_scalar('Base/Total_Loss', losses, epoch)
        writer.add_scalar('Base/Loss_Exp', loss_exp, epoch)

        if epoch > 0 and (epoch % 10 == 0 or epoch == base_epochs - 1):
            eval_metrics = test(config, test_loader, model, device=device, epoch=epoch)
            
            current_total_test_loss = eval_metrics['l1_exp'] + eval_metrics['l1_pose']
            if current_total_test_loss < best_base_loss:
                best_base_loss = current_total_test_loss
                print(f"+++ New best base test loss: {best_base_loss:.4f}. Saving...")
                if not getattr(args, 'disable_save', False):
                    torch.save(model.state_dict(), osp.join(ckpt_folder, 'realm_best_base.pt'))
                    
            torch.save(model.state_dict(), osp.join(ckpt_folder, 'realm_latest_base.pt'))


    # ==========================================
    # STAGE 2: REFINEMENT MODULE TRAINING
    # ==========================================
    print(f"--- Starting Refinement Module Training ({refine_epochs} Epochs) ---")
    best_refine_loss = float('inf')
    total_refine_steps = len(train_loader) * refine_epochs
    warmup_refine_steps = int(total_refine_steps * warmup_ratio)
    
    base_ckpt_path = osp.join(ckpt_folder, 'realm_best_base.pt')
   
    print(f"Loading frozen base weights from: {base_ckpt_path}")
    resume = torch.load(base_ckpt_path, map_location=device, weights_only=True)
    
    state_dict = model.state_dict() 
    for k, v in state_dict.items(): 
        state_dict[k] = resume.get(k, v)
    model.load_state_dict(state_dict)

    # Freeze base parameters, keep refine_net trainable
    for name, param in model.named_parameters():
        if 'refine_net' not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    discriminator = TemporalDiscriminator(motion_dim=config.model.refine_dim).to(device) 
    adversarial_loss = nn.BCEWithLogitsLoss().to(device)

    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=config.train.lr * 0.25, betas=(0.5, 0.999))
    optimizer_G = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.train.lr, weight_decay=1e-4)

    scheduler_G = get_lr_scheduler(optimizer_G, config.train.lr_policy, warmup_steps=warmup_refine_steps, total_steps=total_refine_steps, **config.train.step_scheduler_params)
    scheduler_D = get_lr_scheduler(optimizer_D, config.train.lr_policy, warmup_steps=warmup_refine_steps, total_steps=total_refine_steps, **config.train.step_scheduler_params)

    for epoch in range(refine_epochs):
        g_total, g_l1, g_adv, d_total = train_one_epoch_refine(
            config, train_loader, model, discriminator, optimizer_G, optimizer_D, 
            scheduler_G, scheduler_D, device, epoch, adversarial_loss
        )

        writer.add_scalar('Refine/G_Total', g_total, epoch)
        writer.add_scalar('Refine/G_L1', g_l1, epoch)
        writer.add_scalar('Refine/G_Adv', g_adv, epoch)
        writer.add_scalar('Refine/D_Total', d_total, epoch)

        if epoch > 0 and (epoch % 10 == 0 or epoch == refine_epochs - 1):
            eval_metrics = test(config, test_loader, model, device=device, epoch=epoch)
            
            current_total_test_loss = eval_metrics['fid_delta_exp'] 

            if current_total_test_loss < best_refine_loss:
                best_refine_loss = current_total_test_loss
                print(f"+++ New best refine test loss: {best_refine_loss:.4f}. Saving...")
                if not getattr(args, 'disable_save', False):
                    torch.save(model.state_dict(), osp.join(ckpt_folder, 'realm_best_refine.pt'))
                    
            torch.save(model.state_dict(), osp.join(ckpt_folder, 'realm_latest_refine.pt'))

print("--- Training Complete ---")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    # ADDED 'both' to choices
    parser.add_argument('--stage', type=str, default='both', choices=['base', 'refine', 'both'], help='Training stage to run')
    parser.add_argument('--trial', type=str, default='default', help='Trial name')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID')
    parser.add_argument('--ckpt_folder', type=str, default='', help='Path to load base checkpoint from')
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--disable_save', action='store_true')
    
    args = parser.parse_args()
    
    CUR_DIR = os.path.dirname(os.path.abspath(__file__))
    main(args)