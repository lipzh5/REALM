import numpy as np
import scipy


def calculate_rpcc_global(pred_seq, gt_seq, spk_seq): # flattened features
    # pred_seq = flatten_features(pred_seq_list)
    # gt_seq = flatten_features(gt_seq_list)
    # spk_seq = flatten_features(spk_seq_list)
    # print(f'pred seq: {pred_seq.shape}, spc seq: {spk_seq}')
    pred_pcc = np.corrcoef(pred_seq.reshape(-1, ), spk_seq.reshape(-1, ))[0, 1]
    gt_pcc = np.corrcoef(gt_seq.reshape(-1, ), spk_seq.reshape(-1, ))[0, 1]
    rpcc = abs(pred_pcc - gt_pcc)
    # print(f'rpcc: {rpcc}')
    return rpcc

def calculate_frechet_distance(pred_features, gt_features, eps=1e-6):
    """Standard global Fréchet Distance."""
    mu1, sigma1 = np.mean(pred_features, axis=0), np.cov(pred_features, rowvar=False)
    mu2, sigma2 = np.mean(gt_features, axis=0), np.cov(gt_features, rowvar=False)
    diff = mu1 - mu2
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    tr_covmean = np.trace(covmean)
    return (diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)

def calculate_fid_fm(pred_seq_list, gt_seq_list):
    """Calculates FID_fm: The average FID score across K individual videos."""
    fid_scores = []
    for pred_beta, gt_beta in zip(pred_seq_list, gt_seq_list):
        if pred_beta.shape[0] < 2 or gt_beta.shape[0] < 2: continue
        try:
            fid_scores.append(calculate_frechet_distance(pred_beta, gt_beta))
        except ValueError:
            continue
    return np.mean(fid_scores) if len(fid_scores) > 0 else 0.0

def calculate_fid_delta(pred_seq_list, gt_seq_list):
    """Calculates FID_Delta fm on the temporal differences of lists of sequences."""
    diff_pred_list = [np.diff(p, axis=0) for p in pred_seq_list]
    diff_gt_list = [np.diff(g, axis=0) for g in gt_seq_list]
    return calculate_fid_fm(diff_pred_list, diff_gt_list)

def calculate_paired_fid_fm(pred_seq_list, gt_seq_list, spk_seq_list):
    """Calculates FID_fm: The average FID score across K individual videos."""
    fid_scores = []
    for spk_beta, pred_beta, gt_beta in zip(spk_seq_list, pred_seq_list, gt_seq_list):
        if pred_beta.shape[0] < 2 or gt_beta.shape[0] < 2 or spk_beta.shape[0] < 2: continue
        try:
            cat_pred = np.concatenate([spk_beta, pred_beta], axis=-1)
            cat_gt = np.concatenate([spk_beta, gt_beta], axis=-1)
            fid_scores.append(calculate_frechet_distance(cat_pred, cat_gt))
        except ValueError:
            continue
    return np.mean(fid_scores) if len(fid_scores) > 0 else 0.0

def flatten_features(seq_list):
    """Safely flattens a list of variable-length sequences for global FD."""
    return np.concatenate([seq.reshape(-1, seq.shape[-1]) for seq in seq_list], axis=0)
