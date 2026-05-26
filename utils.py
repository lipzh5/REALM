import os
from scipy import linalg
import numpy as np
import torch
from torch.optim import lr_scheduler
from transformers import get_cosine_schedule_with_warmup
import yaml
import json

import matplotlib
# Use the non-interactive Agg backend to avoid Tkinter thread issues
matplotlib.use('Agg') 
import matplotlib.pyplot as plt


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


NUM_BINS = 10
BIN_EDGES = torch.linspace(0.0, 1.0, NUM_BINS + 1)  # [0.0, 0.1, ..., 1.0]
def discretize_blendshapes(bs, num_bins=10):
    """
    bs: (B, 52) continuous blendshape coefficients in [0, 1]
    return: (B, 52) integer labels in [0, num_bins-1]
    """
    bs = torch.clamp(bs, 0.0, 1.0)
    labels = torch.floor(bs * num_bins).long()
    labels = torch.clamp(labels, max=num_bins - 1)
    return labels


def undiscretize_blendshapes(labels, num_bins=10):
    """
    labels: (B, 52) integer labels
    return: (B, 52) float blendshapes in [0, 1]
    """
    return (labels.float() + 0.5) / num_bins




def log_losses_jsonl(filepath, **loss_dict):
    with open(filepath, "a") as f:
        f.write(json.dumps(loss_dict) + "\n")


def get_model_size(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return size_all_mb



def pad_to(seq: torch.Tensor, length: int) -> torch.Tensor:
    L = seq.shape[0]
    if L < length:
        pad_shape = (length - L, *seq.shape[1:])
        return torch.cat([seq, seq.new_zeros(pad_shape)], dim=0)
    return seq


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)
    

def prepare_sub_folder(output_directory):
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory, exist_ok=True)
    return checkpoint_directory


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class DictAverageMeter:
    def __init__(self, *keys):
        self.meters = {key: AverageMeter() for key in keys}

    def __getitem__(self, key):
        if key in self.meters:
            return self.meters[key]
        raise AttributeError("has no attribute '{}'".format(key))

    def __getattr__(self, key):
        if key in self.meters:
            return self.meters[key]
        raise AttributeError("has no attribute '{}'".format(key))

    def update(self, val_dict):
        for key, values in val_dict.items():
            self.meters[key].update(values['val'], values.get('n', 1))

    def reset(self):
        for key in self.meters:
            self.meters[key].reset()

def get_lr_scheduler(optimizer, lr_policy, **kwargs):
    if lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=kwargs['step_size'],
                                        gamma=kwargs['gamma'], last_epoch=kwargs['last_epoch'])
    elif lr_policy == 'cosine':
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=kwargs['warmup_steps'],
                                        num_training_steps=kwargs['total_steps'])
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', lr_policy)
    return scheduler
       

def get_scheduler(optimizer, hyperparameters, iterations=-1):
    """for vico challenge baseline"""
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(0, 2, 3, 1)


def torch_img_to_np2(img):
    img = img.detach().cpu().numpy()
    # img = img * np.array([0.229, 0.224, 0.225]).reshape(1,-1,1,1)
    # img = img + np.array([0.485, 0.456, 0.406]).reshape(1,-1,1,1)
    img = img * np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img + np.array([0.5, 0.5, 0.5]).reshape(1, -1, 1, 1)
    img = img.transpose(0, 2, 3, 1)
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)[:, :, :, [2, 1, 0]]

    return img

import torch.nn.functional as F
import numpy as np

def torch_img_to_np3(img, size=224):
    """
    img: (B, C, H, W), normalized to [-1, 1]
    """

    # 🔹 Resize first (still float tensor)
    img = F.interpolate(
        img,
        size=(size, size),
        mode="bilinear",
        align_corners=False
    )

    # 🔹 De-normalize [-1, 1] → [0, 1]
    img = img * 0.5 + 0.5

    # 🔹 Torch → NumPy
    img = img.detach().cpu().numpy()
    img = img.transpose(0, 2, 3, 1)  # (B, H, W, C)

    # 🔹 [0,1] → [0,255]
    img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 🔹 RGB → BGR (OpenCV-style, optional)
    img = img[:, :, :, [2, 1, 0]]

    return img



# from datasets.react25 import ReactionDataset
# from datasets.vico import ViCoDataset
# from torch.utils.data import DataLoader
# def get_data_loader(config, split, rendering=False):
#     if config.data.dataset == 'react':
#         dataset = ReactionDataset(root_dir=config.data.data_root, split=split, for_rendering=rendering, **config.data.react)
#     else:
#         dataset = ViCoDataset(root_dir=config.data.data_root, split=split, **config.data.vico)
#     batch_size = 1 if split == 'test' else config.train.batch_size
#     data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=config.train.num_workers, shuffle=True if split=='train' else False)

#     return data_loader


def split_coeff(coeffs): # for deep3d reconstruction
    id_coeffs = coeffs[:, :80]
    exp_coeffs = coeffs[:, 80: 144]
    tex_coeffs = coeffs[:, 144: 224]
    angles = coeffs[:, 224: 227]
    gammas = coeffs[:, 227: 254]
    translations = coeffs[:, 254:]
    return {
        'id': id_coeffs,
        'exp': exp_coeffs,
        'tex': tex_coeffs,
        'angle': angles,
        'gamma': gammas,
        'trans': translations
    }




def plot_speaker_vs_listener_gt(speaker_motion_gt, listener_motion_gt, feature_idx=0):
    """
    Plots a specific dimension of the speaker's GT motion against the listener's GT motion.
    
    Args:
        speaker_motion_gt: Tensor of shape (bs, T, D)
        listener_motion_gt: Tensor of shape (bs, T, D)
        feature_idx: The specific feature dimension to plot (0 to D-1)
        batch_idx: Which sequence in the batch to visualize
    """
    # Detach from graph and move to CPU for plotting
    speaker_seq = speaker_motion_gt[:, feature_idx]
    listener_seq = listener_motion_gt[:, feature_idx]
    
    time_steps = range(len(speaker_seq))

    plt.figure(figsize=(12, 5))
    
    # Plot Speaker GT
    plt.plot(time_steps, speaker_seq, label='Speaker GT', color='blue', alpha=0.8, linewidth=1.5)
    
    # Plot Listener GT
    plt.plot(time_steps, listener_seq, label='Listener GT', color='orange', alpha=0.8, linewidth=1.5)
    
    plt.title(f'Speaker GT vs Listener GT (Feature Dimension: {feature_idx})')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Feature Value')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    
    # Save or show
    plt.savefig('speaker_vs_listener_gt.pdf')
    plt.show()
    plt.close()


def debug_plot_rot_prediction(g_rots, p_rots, save_path):
    fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    titles = ['Pitch', 'Yaw', 'Roll']
    colors = ['r', 'g', 'b']
    
    for i in range(3):
        # We plot the full sequence to see the GT initialization blend into the prediction
        axs[i].plot(g_rots[:, i], label='Ground Truth', color='black', linestyle='--')
        axs[i].plot(p_rots[:, i], label='Predicted', color=colors[i])
        axs[i].axvline(x=0, color='gray', linestyle=':', label='Generation Start')
        axs[i].set_ylabel('Intensity')
        axs[i].set_title(titles[i])
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
    
    axs[2].set_xlabel('Frames')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)
    # plt.show()

def debug_plot_prediction(gt, pred, save_path):
    nplots = gt.shape[-1]
    fig, axs = plt.subplots(nplots, 1, figsize=(12, 10), sharex=True)
    # titles = ['Pitch', 'Yaw', 'Roll']
    colors = ['r', 'g', 'b']
    
    for i in range(nplots):
        # We plot the full sequence to see the GT initialization blend into the prediction
        axs[i].plot(gt[:, i], label='Ground Truth', color='black', linestyle='--')
        axs[i].plot(pred[:, i], label='Predicted', color=colors[i%3])
        axs[i].axvline(x=0, color='gray', linestyle=':', label='Generation Start')
        axs[i].set_ylabel('Intensity')
        axs[i].set_title(f'dim {i}')
        axs[i].legend(loc='upper right')
        axs[i].grid(True, alpha=0.3)
    
    axs[2].set_xlabel('Frames')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    # plt.show()
    plt.clf()
    plt.close(fig)


def debug_plot_rot_prediction_frequency(g_rots, p_rots, save_path, fps=30):
    # Create a 3x2 grid: Left for Time Domain, Right for Frequency Domain
    fig, axs = plt.subplots(3, 2, figsize=(18, 10))
    titles = ['Pitch', 'Yaw', 'Roll']
    colors = ['r', 'g', 'b']
    
    # Calculate frequency bins based on sequence length and framerate
    N = len(g_rots)
    freqs = np.fft.rfftfreq(N, d=1/fps)
    
    for i in range(3):
        # --- Time Domain (Left Column) ---
        axs[i, 0].plot(g_rots[:, i], label='Ground Truth', color='black', linestyle='--')
        axs[i, 0].plot(p_rots[:, i], label='Predicted', color=colors[i])
        axs[i, 0].axvline(x=0, color='gray', linestyle=':', label='Generation Start')
        axs[i, 0].set_ylabel('Intensity')
        axs[i, 0].set_title(f'{titles[i]} - Time Domain')
        axs[i, 0].legend(loc='upper right')
        axs[i, 0].grid(True, alpha=0.3)
        
        # --- Frequency Domain (Right Column) ---
        # Compute magnitude of the Real FFT
        fft_g = np.abs(np.fft.rfft(g_rots[:, i]))
        fft_p = np.abs(np.fft.rfft(p_rots[:, i]))
        
        # Normalize the magnitudes by sequence length for fair scaling
        fft_g = fft_g / N
        fft_p = fft_p / N
        
        axs[i, 1].plot(freqs, fft_g, label='Ground Truth', color='black', linestyle='--')
        axs[i, 1].plot(freqs, fft_p, label='Predicted', color=colors[i])
        axs[i, 1].set_ylabel('Magnitude')
        axs[i, 1].set_title(f'{titles[i]} - Frequency Domain')
        axs[i, 1].legend(loc='upper right')
        axs[i, 1].grid(True, alpha=0.3)
        
        # Optional: Uncomment the next line if high frequencies are too compressed to see
        # axs[i, 1].set_yscale('log') 
    
    # Set x-axis labels for the bottom row
    axs[2, 0].set_xlabel('Frames')
    axs[2, 1].set_xlabel('Frequency (Hz)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path)
    plt.clf()
    plt.close(fig)



