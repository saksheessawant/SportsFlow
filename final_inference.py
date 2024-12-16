import os
import torch
import math
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import cv2
import numpy as np
from train_log.RIFE_HDv3 import Model
from lpips import LPIPS
from skimage.metrics import structural_similarity as compare_ssim

def compute_ssim(img1, img2):
    """
    Compute SSIM for two images.
    Args:
        img1 (torch.Tensor): First image.
        img2 (torch.Tensor): Second image.
    Returns:
        float: SSIM value.
    """
    img1_np = img1.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format
    img2_np = img2.squeeze(0).permute(1, 2, 0).cpu().numpy()  # Convert to HWC format

    # Ensure images are at least 7x7 or smaller than the specified win_size
    win_size = min(7, img1_np.shape[0], img1_np.shape[1])
    
    # Specify data_range based on the pixel value range
    data_range = img1_np.max() - img1_np.min()
    
    ssim, _ = compare_ssim(
        img1_np, img2_np, full=True, multichannel=True, win_size=win_size, channel_axis=-1, data_range=data_range
    )
    return ssim



# Evaluation Function with additional metrics
def evaluate(model, val_loader, device, output_file='evaluation_results_humanloss_epoch4.txt'):
    model.eval()
    loss_l1_list = []
    psnr_list = []
    lpips_list = []
    ssim_list = []

    lpips_loss = LPIPS(net='alex').to(device)  # Load LPIPS model

    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(val_loader):
            data = data.to(device)
            gt = gt.to(device)

            # Split concatenated inputs
            img0, img1 = data[:, :3], data[:, 3:]

            # Perform inference
            pred = model.inference(img0, img1)

            # Calculate L1 Loss
            loss_l1 = torch.abs(pred - gt).mean().item()
            loss_l1_list.append(loss_l1)

            # Calculate PSNR
            mse = torch.mean((pred - gt) ** 2).item()
            psnr = -10 * math.log10(mse) if mse > 0 else float('inf')
            psnr_list.append(psnr)

            # Calculate LPIPS
            lpips_val = lpips_loss(pred, gt).mean().item()
            lpips_list.append(lpips_val)

            # Calculate SSIM
            for i in range(pred.size(0)):
                ssim_val = compute_ssim(pred[i].unsqueeze(0), gt[i].unsqueeze(0))
                ssim_list.append(ssim_val)

            # Print intermediate results for this batch
            print(f"Batch {batch_idx + 1} - L1 Loss: {loss_l1:.4f}, PSNR: {psnr:.4f} dB, LPIPS: {lpips_val:.4f}, SSIM: {np.mean(ssim_list):.4f}")

    # Compute averages
    avg_loss_l1 = np.mean(loss_l1_list)
    avg_psnr = np.mean(psnr_list)
    avg_lpips = np.mean(lpips_list)
    avg_ssim = np.mean(ssim_list)

    # Save results
    with open(output_file, 'w') as f:
        f.write(f"Average L1 Loss: {avg_loss_l1:.4f}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f} dB\n")
        f.write(f"Average LPIPS: {avg_lpips:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
    print(f"Evaluation complete. Results saved to {output_file}")


# Custom Dataset for Triplets
class TripletDataset(Dataset):
    def __init__(self, root_dir, start_clip=6235, end_clip=6285):
        """
        Args:
            root_dir (str): Path to the folder containing triplets.
            start_clip (int): Starting clip number (inclusive).
            end_clip (int): Ending clip number (inclusive).
        """
        self.root_dir = root_dir
        self.start_clip = start_clip
        self.end_clip = end_clip
        self.samples = self._filter_clips()

    def _filter_clips(self):
        filtered_samples = []
        for sample in sorted(os.listdir(self.root_dir)):
            clip_num = int(sample.split('_')[1])  # Assumes format: clip_XXXX_triplet_YYYY
            if self.start_clip <= clip_num <= self.end_clip:
                filtered_samples.append(sample)
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        triplet_dir = os.path.join(self.root_dir, self.samples[idx])
        img0 = cv2.imread(os.path.join(triplet_dir, "img0.jpg"))
        img1 = cv2.imread(os.path.join(triplet_dir, "img1.jpg"))
        gt = cv2.imread(os.path.join(triplet_dir, "gt.jpg"))

        # Convert images to tensors
        img0 = ToTensor()(img0)
        img1 = ToTensor()(img1)
        gt = ToTensor()(gt)

        # Concatenate img0 and img1 as input
        data = torch.cat((img0, img1), dim=0)
        return data, gt





# Main Code
if __name__ == "__main__":
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model()
    state_dict = torch.load('finall_acv_model_epoch_4.pth', map_location=device)
    model.flownet.load_state_dict(state_dict)


    # # Load model
    # state_dict = torch.load('finall_acv_model_epoch_1.pth', map_location=device)
        
    # if isinstance(state_dict, dict):  # Ensure it's a state_dict
    #     model.load_state_dict(state_dict)
    # else:  # If it isn't, assume it's the full model object
    #     model = state_dict

    model.eval()
    model.device()
    print("Fine-tuned model loaded.")
    # Load model
    # model = Model()
    # model.load_model('train_log', -1)  # Adjust this path if needed
    model.eval()
    model.device()
    print("Model loaded.")

    # Dataset and DataLoader
    dataset = TripletDataset('./ValidationTriplets', start_clip=6235, end_clip=6285)
    val_loader = DataLoader(dataset, batch_size=4, shuffle=False)

    # Evaluate the model
    evaluate(model, val_loader, device)
