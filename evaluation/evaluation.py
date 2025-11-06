import os
import cv2
import sys
import piq
import math
import torch
import argparse
import warnings
import numpy as np
import torchvision.transforms as transforms

from tqdm import tqdm
from pygame import freetype
from pytorch_fid import fid_score
from scipy import signal, ndimage
from scipy.ndimage import gaussian_filter
from PIL import Image, ImageFont, ImageDraw
from torchvision.transforms import Compose, ToTensor, Resize
from skimage.metrics import structural_similarity as skimage_ssim

from utils import devdata, fspecial_gauss, render_normal, perspective, center2size

freetype.init()
warnings.filterwarnings("ignore", message="nn.functional.upsample is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")

# Add prestyle directory to path for importing StyleNet related modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'prestyle'))
from trainer import StyleNet

RESIZE = transforms.Resize((128, 128), transforms.InterpolationMode.BICUBIC, antialias=True)
TRANSFER = transforms.ToTensor()


def get_target_text_dict(gt_txt_path):
    with open(gt_txt_path, 'r', encoding='utf-8') as f:
        target_text_list = [line.strip() for line in f.readlines()]
        target_text_dict = {line.split(' ')[0]: line.split(' ')[1] for line in target_text_list}
    return target_text_dict

def make_color_target(target_text, template_font):
    global RESIZE, TRANSFER

    font = freetype.Font(template_font)
    font.antialiased = True
    font.origin = True
    font.size = 40
    font.underline = False
    font.strong = False
    font.oblique = False

    color_target, _ = render_normal(font, target_text)

    rotate = 0
    zoom = (1, 1)
    shear = (0, 0)
    perspect = (0, 0)
    padding = (10, 10, 10, 10)

    color_target = perspective(color_target, rotate, zoom, shear, perspect, padding)
    
    h, w = color_target.shape[:2]
    color_target = center2size(color_target, (h, w))

    color_target = Image.fromarray(color_target).convert('RGB')
    color_target = RESIZE(color_target)
    color_target = TRANSFER(color_target)
    color_target = color_target.unsqueeze(0)

    return color_target

def make_font_template(target_text, template_font):
    template_font = ImageFont.truetype(template_font, 64)
    std_l, std_t, std_r, std_b = template_font.getbbox(target_text)
    std_h, std_w = std_b - std_t, std_r - std_l
    img_font_template = Image.new('RGB', (std_w + 10, std_h + 10), color=(0, 0, 0))
    draw = ImageDraw.Draw(img_font_template)
    draw.text((5, 5), target_text, fill=(255, 255, 255), font=template_font, anchor="lt")
    img_font_template = RESIZE(img_font_template)
    array_font_template = np.array(img_font_template)
    edge_font_template = cv2.Canny(array_font_template, 100, 200)
    font_template = TRANSFER(edge_font_template)
    font_template = font_template.repeat(3, 1, 1)
    font_template = font_template.unsqueeze(0)

    return font_template

def extract_embeddings(model, image_path, target_text, template_font, device='cuda', image_size=128):

    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    input_image = transform(image).unsqueeze(0)
    input_image = input_image.to(device)

    color_target = make_color_target(target_text, template_font).to(device)
    font_template = make_font_template(target_text, template_font).to(device)
    
    with torch.no_grad():
        img_spatial, img_glyph = model(input_image)
        
        out_removal, out_seg = model.spatial_head(img_spatial)
        out_color, out_font = model.glyph_head(img_glyph, color_target, font_template)
        
        results = {
            'spatial_embedding': img_spatial.cpu(),
            'glyph_embedding': img_glyph.cpu(),
            'out_removal': out_removal.cpu(),
            'out_seg': out_seg.cpu(),
            'out_color': out_color.cpu(),
            'out_font': out_font.cpu(),
            'color_target': color_target.cpu()
        }

    return results

def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(float)
    img2 = img2.astype(float)

    size = min(img1.shape[0], 11)
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.fftconvolve(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on
    Signals, Systems and Computers, Nov. 2003

    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2)) / 4.0
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(img1, img2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.convolve(img1, downsample_filter, mode='reflect')
        filtered_im2 = ndimage.convolve(img2, downsample_filter, mode='reflect')
        im1 = filtered_im1[:: 2, :: 2]
        im2 = filtered_im2[:: 2, :: 2]
    # Note: Remove the negative and add it later to avoid NaN in exponential.
    sign_mcs = np.sign(mcs[0: level - 1])
    sign_mssim = np.sign(mssim[level - 1])
    mcs_power = np.power(np.abs(mcs[0: level - 1]), weight[0: level - 1])
    mssim_power = np.power(np.abs(mssim[level - 1]), weight[level - 1])
    return np.prod(sign_mcs * mcs_power) * sign_mssim * mssim_power

def compute_cided2000(img1, img2, normalize=False, mask=None, delta_e_max=50):
    """
    Compute the CIEDE2000 color difference between two images.
    If normalize=True, return similarity in [0,1] via 1 - (mean_deltaE / delta_e_max).
    """
    import numpy as np
    from skimage.color import rgb2lab, deltaE_ciede2000
    import torch

    # Convert tensors to numpy arrays
    def to_numpy(img):
        if torch.is_tensor(img):
            img = img.cpu().numpy()
        return img

    arr1 = to_numpy(img1)
    arr2 = to_numpy(img2)
    # Squeeze batch and reorder to HWC
    if arr1.ndim == 4:
        arr1 = arr1.squeeze(0)
    if arr2.ndim == 4:
        arr2 = arr2.squeeze(0)
    # Channel first to last
    if arr1.shape[0] == 3:
        arr1 = np.transpose(arr1, (1, 2, 0))
    if arr2.shape[0] == 3:
        arr2 = np.transpose(arr2, (1, 2, 0))
    # Scale [0,1] to [0,255]
    if arr1.max() <= 1.0:
        arr1 = (arr1 * 255).astype(np.uint8)
    else:
        arr1 = arr1.astype(np.uint8)
    if arr2.max() <= 1.0:
        arr2 = (arr2 * 255).astype(np.uint8)
    else:
        arr2 = arr2.astype(np.uint8)

    # Convert RGB to Lab
    lab1 = rgb2lab(arr1)
    lab2 = rgb2lab(arr2)
    # Compute deltaE per pixel
    delta = deltaE_ciede2000(lab1, lab2)
    # Apply mask if provided
    if mask is not None:
        delta = delta[mask]
    # Mean deltaE
    mean_delta = float(np.mean(delta))
    if normalize:
        sim = 1.0 - min(mean_delta / delta_e_max, 1.0)
        return sim
    return mean_delta

def compute_fsim(img1, img2):
    """
    Compute Feature Similarity Index (FSIM) for two images.
    """
    # Convert numpy to tensor
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1)
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2)
    
    # Ensure batch dimension
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)
    
    # Move to device and float
    img1 = img1.to(DEVICE).float()
    img2 = img2.to(DEVICE).float()
    # Clamp values to [0,1] to satisfy piq.fsim requirements
    img1 = img1.clamp(0.0, 1.0)
    img2 = img2.clamp(0.0, 1.0)
    
    # Compute FSIM using piq library
    value = piq.fsim(img1, img2, data_range=1.0)
    
    # Normalize if required
    return value.item()

def compute_msssim(img1, img2, levels=5, weights=None):
    if weights is None:
        weights = np.array([0.0448,0.2856,0.3001,0.2363,0.1333])
    
    # Convert tensors to numpy arrays
    if torch.is_tensor(img1):
        img1 = img1.cpu().numpy()
    if torch.is_tensor(img2):
        img2 = img2.cpu().numpy()
    
    # Squeeze batch dimension if present
    if img1.ndim == 4:
        img1 = img1.squeeze(0)
    if img2.ndim == 4:
        img2 = img2.squeeze(0)
    
    # Channel first to last
    if img1.shape[0] in (1, 3):
        img1 = np.transpose(img1, (1, 2, 0))
    if img2.shape[0] in (1, 3):
        img2 = np.transpose(img2, (1, 2, 0))
    
    # Ensure float32 type
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Check minimum size requirement
    min_size = 11  # minimum window size for SSIM
    if img1.shape[0] < min_size or img1.shape[1] < min_size:
        # If image is too small, return fallback SSIM
        try:
            return float(skimage_ssim(img1, img2, data_range=1.0, channel_axis=-1))
        except:
            return 0.0
    
    mssim_vals, mcs_vals = [], []

    for l in range(levels):
        try:
            # Check if image is still large enough
            if img1.shape[0] < min_size or img1.shape[1] < min_size:
                break
                
            ssim_map, cs_map = skimage_ssim(img1, img2, full=True, data_range=1.0, channel_axis=-1)
            
            # Check for NaN or invalid values
            if np.isnan(ssim_map).any() or np.isnan(cs_map).any():
                break
                
            mssim_vals.append(float(ssim_map.mean()))
            mcs_vals.append(float(cs_map.mean()))

            # Gaussian filter + downsample
            img1 = gaussian_filter(img1, sigma=1)[::2, ::2]
            img2 = gaussian_filter(img2, sigma=1)[::2, ::2]
            
        except Exception as e:
            # If any error occurs, break the loop
            break

    # Check if we have enough valid measurements
    if len(mssim_vals) == 0:
        return 0.0
    
    # Adjust levels and weights based on actual number of measurements
    actual_levels = len(mssim_vals)
    if actual_levels < levels:
        weights = weights[:actual_levels]
        weights = weights / weights.sum()  # renormalize
    
    mssim_vals = np.array(mssim_vals)
    mcs_vals = np.array(mcs_vals)

    # Check for any NaN or invalid values
    if np.isnan(mssim_vals).any() or np.isnan(mcs_vals).any():
        return 0.0
    
    # Ensure all values are positive for power calculation
    mcs_vals = np.abs(mcs_vals)
    mssim_vals = np.abs(mssim_vals)
    
    # Add small epsilon to avoid zero values
    epsilon = 1e-10
    mcs_vals = np.maximum(mcs_vals, epsilon)
    mssim_vals = np.maximum(mssim_vals, epsilon)

    try:
        if actual_levels == 1:
            overall = mssim_vals[0]
        else:
            overall = np.prod(mcs_vals[:-1]**weights[:-1]) * (mssim_vals[-1]**weights[-1])
        
        # Check if result is valid
        if np.isnan(overall) or np.isinf(overall):
            return 0.0
        
        return float(overall)
        
    except Exception as e:
        return 0.0


def compute_text_appearance_similarity(sim_color, sim_font, sim_removal):

    text_style_similarity = (sim_color + sim_font + sim_removal) / 3.0
    
    return text_style_similarity


def calculate_metrics_for_images(img_path, gt_path, target_text, template_font):

    global MODEL, DEVICE

    transform = Compose([
        Resize(size=(128, 128)),
        ToTensor(),
    ])
    
    img = Image.open(img_path).convert('RGB')
    gt = Image.open(gt_path).convert('RGB')
    
    img_tensor = transform(img).unsqueeze(0)
    gt_tensor = transform(gt).unsqueeze(0)
    
    # MSE
    mse = ((gt_tensor - img_tensor) ** 2).mean().item()
    
    # PSNR
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10 * math.log10(1.0 / mse)
    
    # SSIM
    R = gt_tensor[0, 0, :, :]
    G = gt_tensor[0, 1, :, :]
    B = gt_tensor[0, 2, :, :]
    YGT = .299 * R + .587 * G + .114 * B
    
    R = img_tensor[0, 0, :, :]
    G = img_tensor[0, 1, :, :]
    B = img_tensor[0, 2, :, :]
    YBC = .299 * R + .587 * G + .114 * B
    
    ssim_value = msssim(np.array(YGT * 255), np.array(YBC * 255))

    results_img = extract_embeddings(MODEL, img_path, target_text, template_font, DEVICE)
    results_gt = extract_embeddings(MODEL, gt_path, target_text, template_font, DEVICE)

    out_color_img = results_img['out_color']
    out_color_gt = results_gt['out_color']
    color_target = results_img['color_target']

    mask = color_target.squeeze(0).mean(dim=0) > 0
    mask = mask.cpu().numpy()
    cided2000_color_masked_norm = compute_cided2000(out_color_img, out_color_gt, normalize=True, mask=mask, delta_e_max=50)
    
    out_font_img = results_img['out_font']
    out_font_gt = results_gt['out_font']
    fsim_font = compute_fsim(out_font_img, out_font_gt)

    out_removal_img = results_img['out_removal']
    out_removal_gt = results_gt['out_removal']
    msssim_removal = compute_msssim(out_removal_img, out_removal_gt)

    tas_score = compute_text_appearance_similarity(cided2000_color_masked_norm, fsim_font, msssim_removal)
    
    metrics = {
        'ssim': float(ssim_value),
        'psnr': float(psnr),
        'mse': float(mse),
        's_color': float(cided2000_color_masked_norm),
        's_font': float(fsim_font),
        's_bg': float(msssim_removal),
        'tas_score': float(tas_score),
        'img_path': img_path,
        'gt_path': gt_path,
    }
    
    return metrics


from utils import load_config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Calculate 7 metrics for image comparison')
    parser.add_argument('--config', type=str, required=True, help='Config file name')
    parser.add_argument('--lang', type=str, required=True)
    
    args = parser.parse_args()
    config_name = args.config
    lang = args.lang
    
    cfg = load_config(config_name)

    img_path = cfg.target_path
    gt_path = cfg.gt_path
    gt_txt_path = cfg.gt_txt_path
    results_dir = cfg.results_dir
    checkpoint_path = cfg.checkpoint_path

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MODEL = StyleNet.load_from_checkpoint(checkpoint_path, map_location=DEVICE)
    MODEL.eval()
    MODEL.to(DEVICE)
    
    os.makedirs(results_dir, exist_ok=True)
    
    img_files = [f for f in os.listdir(img_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    gt_files = [f for f in os.listdir(gt_path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    common_files = sorted([f for f in img_files if f in gt_files])
    target_text_dict = get_target_text_dict(gt_txt_path)
    
    print(f"Found {len(common_files)} matching images to evaluate.")

    sum_metrics = {
        'ssim': 0.0,
        'psnr': 0.0,
        'mse': 0.0,
        's_color': 0.0,
        's_font': 0.0,
        's_bg': 0.0,
        'tas_score': 0.0
    }
    all_results = []
    
    if lang == 'ko':
        template_font = '../datagen/fonts/ko_ttf/NanumGothic.ttf'
    if lang == 'ar':
        template_font = '../datagen/fonts/ar_ttf/Noto_Sans_Arabic.ttf'
    if lang == 'jp':
        template_font = '../datagen/fonts/jp_ttf/Kozuka_Gothic_Pro_R.ttf'
    torch.cuda.empty_cache()

    for filename in tqdm(common_files):
        img_file_path = os.path.join(img_path, filename)
        gt_file_path = os.path.join(gt_path, filename)
        target_text = target_text_dict[filename]
        
        metrics = calculate_metrics_for_images(img_file_path, gt_file_path, target_text, template_font)
        
        for key in sum_metrics.keys():
            sum_metrics[key] += metrics[key]
        
        metrics['fid'] = None
        all_results.append(metrics)
    
    print("Calculating FID score...")
    batch_size = 1

    dims = 2048
    try:
        fid_value = fid_score.calculate_fid_given_paths([str(gt_path), str(img_path)], batch_size, DEVICE, dims)
    except Exception as e:
        print(f"Error calculating FID: {e}")
        fid_value = None

    count = len(all_results)
    avg_metrics = {k: v / count for k, v in sum_metrics.items()}
    avg_metrics['fid'] = fid_value
    
    result_file_path = os.path.join(results_dir, f"results_{config_name}.txt")
    with open(result_file_path, 'w') as f:
        f.write(f'SSIM(↑): {round(avg_metrics["ssim"], 4)}\n')
        f.write(f'PSNR(↑): {round(avg_metrics["psnr"], 4)}\n')
        f.write(f'MSE(↓): {round(avg_metrics["mse"], 4)}\n')
        f.write(f'FID(↓): {round(avg_metrics["fid"], 4)}\n')
        f.write(f'Color CIEDE2000 Masked Norm(↑): {round(avg_metrics["s_color"], 4)}\n')
        f.write(f'Font FSIM(↑): {round(avg_metrics["s_font"], 4)}\n')
        f.write(f'Removal MSSSIM(↑): {round(avg_metrics["s_bg"], 4)}\n')
        f.write(f'Text Style Similarity w/FSIM+MSSSIM(↑): {round(avg_metrics["tas_score"], 4)}\n')

    print(f"Average results saved to {result_file_path}")