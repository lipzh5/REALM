import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

# Adjust these imports to match your project structure
from models import ListenerMotionGenerator
from datasets.vico_dataset_fixclip_70d import get_data_loader as get_data_loader_

def get_data_loader(config, split):
    # Force batch size to 1 for clean, sequential generation and saving
    config.train.batch_size = 1
    return get_data_loader_(config, split)

def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load Config and enforce test settings
    config = OmegaConf.load(args.config)
    if args.sample_stride:
        config.data.stride = args.sample_stride
    
    # 2. Initialize Data Loader
    print("Loading ViCo Test Set...")
    test_loader = get_data_loader(config, split='test')
    print(f"Found {len(test_loader)} test samples.")

    # 3. Initialize Model
    print("Initializing REALM...")
    model = ListenerMotionGenerator(config, device).to(device)
    model.eval() # Everything in eval mode

    # 4. Load Weights
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint}")
        
    print(f"Loading weights from {args.checkpoint}...")
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    
    # Safely load state dict
    model_state = model.state_dict()
    for k, v in model_state.items():
        model_state[k] = state_dict.get(k, v)
    model.load_state_dict(model_state)

    # 5. Output Directory Setup
    os.makedirs(args.output_dir, exist_ok=True)
    pred_dir = os.path.join(args.output_dir, "predictions")
    gt_dir = os.path.join(args.output_dir, "ground_truth")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)

    print("\n--- Starting ViCo Test Set Generation ---")
    exp_dim = model.exp_dim
    refine_dim = config.model.refine_dim

    with torch.no_grad():
        for bid, batch in enumerate(tqdm(test_loader)):
            speaker_audio = batch['audio'].to(device)
            listener_gt = batch['listener_face'].to(device)
            lengths = batch['lengths'].to(device)
            
            # Fetch a unique ID for saving (fallback to batch index if dataloader doesn't provide one)
            sample_id = batch.get('id', [f"seq_{bid:04d}"])[0]

            # ==========================================
            # Stage 1: Coarse Base Trajectory
            # ==========================================
            pred_coarse, gate_alpha, _, _ = model(speaker_audio, listener_gt, lengths)

            # ==========================================
            # Stage 2: Fine Refinement
            # ==========================================
            refine_input = pred_coarse[:, :, :exp_dim]
            if refine_dim == 67: # If refinement expects exp + trans
                refine_input = torch.cat([refine_input, pred_coarse[:, :, -3:]], dim=-1)

            pred_fine = model.refine_net(refine_input, speaker_audio)

            # ==========================================
            # Stage 3: Merge and Save
            # ==========================================
            # Merge refined expressions with coarse head pose (rotation + translation)
            final_motion = torch.cat([
                pred_fine,                                # Refined facial expressions
                pred_coarse[:, :, exp_dim:exp_dim+3],     # Coarse head rotation
                pred_coarse[:, :, -3:]                    # Coarse head translation
            ], dim=-1)

            # Convert to numpy
            # Assuming batch_size=1 from the dataloader setup
            final_motion_np = final_motion.squeeze(0).cpu().numpy()
            gt_motion_np = listener_gt.squeeze(0).cpu().numpy()
            
            # Determine actual sequence length
            seq_len = lengths[0].item()
            final_motion_np = final_motion_np[:seq_len, :]
            gt_motion_np = gt_motion_np[:seq_len, :]

            # Save arrays
            np.save(os.path.join(pred_dir, f"{sample_id}.npy"), final_motion_np)
            np.save(os.path.join(gt_dir, f"{sample_id}.npy"), gt_motion_np)

            if args.save_gate and gate_alpha is not None:
                gate_dir = os.path.join(args.output_dir, "gates")
                os.makedirs(gate_dir, exist_ok=True)
                gate_np = gate_alpha.squeeze(0).cpu().numpy()[:seq_len, :]
                np.save(os.path.join(gate_dir, f"{sample_id}.npy"), gate_np)

    print(f"\n✅ Generation complete! Outputs saved to: {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="REALM ViCo Test Set Inference")
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained weights (e.g., net_xgen_best_refine.pt)')
    parser.add_argument('--output_dir', type=str, default='results/vico_test', help='Directory to save outputs')
    parser.add_argument('--sample_stride', type=int, default=None, help='Override data stride')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--save_gate', action='store_true', help='Save the dynamic gate activation values')

    args = parser.parse_args()
    main(args)