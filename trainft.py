import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
from train_log.RIFE_HDv3 import Model
from smpler_x import SMPLerX

# Load the pre-trained SMPLer-X model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
smpler_x = SMPLerX.from_pretrained('SMPLer-X-S32')
smpler_x.eval()
smpler_x.to(device)

# Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, root_dir, start_clip=6235, end_clip=6285):
        self.root_dir = root_dir
        self.start_clip = start_clip
        self.end_clip = end_clip
        self.samples = self._filter_clips()

    def _filter_clips(self):
        filtered_samples = []
        for sample in sorted(os.listdir(self.root_dir)):
            try:
                clip_num = int(sample.split('_')[1])  # Assumes format: clip_XXXX_triplet_YYYY
                if self.start_clip <= clip_num <= self.end_clip:
                    filtered_samples.append(sample)
            except (IndexError, ValueError):
                print(f"Skipping invalid sample format: {sample}")
        return filtered_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        triplet_dir = os.path.join(self.root_dir, self.samples[idx])
        try:
            img0 = cv2.imread(os.path.join(triplet_dir, "img0.jpg"))
            img1 = cv2.imread(os.path.join(triplet_dir, "img1.jpg"))
            gt = cv2.imread(os.path.join(triplet_dir, "gt.jpg"))

            if img0 is None or img1 is None or gt is None:
                raise FileNotFoundError(f"Missing images in {triplet_dir}")

            # Convert images to tensors
            img0 = ToTensor()(img0)
            img1 = ToTensor()(img1)
            gt = ToTensor()(gt)

            # Concatenate img0 and img1 as input
            data = torch.cat((img0, img1), dim=0)
            return data, gt

        except Exception as e:
            print(f"Error loading data for {triplet_dir}: {e}")
            return None


# Function to split dataset into training and validation subsets
def split_dataset(dataset, train_ratio=0.8):
    train_size = int(len(dataset) * train_ratio)
    val_size = len(dataset) - train_size
    return random_split(dataset, [train_size, val_size])


# Freeze early blocks and entry layers, fine-tune the rest
def freeze_layers(model):
    for name, param in model.flownet.named_parameters():
        if "block0" in name or "block1" in name or "block_tea" in name:
            param.requires_grad = False
        elif "conv0" in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    for name, param in model.flownet.named_parameters():
        print(f"{name}: {'Frozen' if not param.requires_grad else 'Trainable'}")

def human_loss(pred_smpl, gt_smpl):
    """
    Calculate the human loss as described in the Humans in 4D paper.
    """
    # SMPL parameter losses
    pose_loss = F.mse_loss(pred_smpl['pose'], gt_smpl['pose'])
    shape_loss = F.mse_loss(pred_smpl['shape'], gt_smpl['shape'])
    
    # 3D joint loss
    joint_loss = F.l1_loss(pred_smpl['joints3d'], gt_smpl['joints3d'])
    
    # 2D joint loss (assuming you have a perspective projection function)
    pred_joints2d = perspective_projection(pred_smpl['joints3d'], pred_smpl['camera'])
    gt_joints2d = perspective_projection(gt_smpl['joints3d'], gt_smpl['camera'])
    joint2d_loss = F.l1_loss(pred_joints2d, gt_joints2d)
    
    # Combine losses
    total_loss = pose_loss + shape_loss + joint_loss + joint2d_loss
    
    return total_loss

class Ternary(torch.nn.Module):
    def __init__(self):
        super(Ternary, self).__init__()
        patch_size = 7
        out_channels = patch_size * patch_size
        # Register as a non-learnable buffer
        self.register_buffer('w', torch.eye(out_channels).reshape((patch_size, patch_size, 1, out_channels)))
        self.w = self.w.permute(3, 2, 0, 1)  # Reshape the weights correctly
    
    def transform(self, img):
        # Convolution using non-learnable weights
        patches = F.conv2d(img, self.w, padding=3, bias=None)
        transf = patches - img  # Subtraction (out-of-place)
        transf_norm = transf / torch.sqrt(0.81 + transf**2)  # Normalize (safe, out-of-place)
        return transf_norm
    
    def forward(self, img0, img1):
        t1 = self.transform(img0)
        t2 = self.transform(img1)
        # Calculate distance
        dist = (t1 - t2) ** 2
        dist_norm = torch.mean(dist / (0.1 + dist), dim=1, keepdim=True)  # Differentiable
        return dist_norm

census = Ternary().to(device)

class LapLoss(torch.nn.Module):
    @staticmethod
    def gauss_kernel(size=5, channels=3):
        kernel = torch.tensor([[1., 4., 6., 4., 1],
                               [4., 16., 24., 16., 4.],
                               [6., 24., 36., 24., 6.],
                               [4., 16., 24., 16., 4.],
                               [1., 4., 6., 4., 1.]])
        kernel /= 256.
        kernel = kernel.repeat(channels, 1, 1, 1)
        kernel = kernel.to(device).requires_grad_(True)
        return kernel


    @staticmethod
    def laplacian_pyramid(img, kernel, max_levels=3):
        def downsample(x):
            return x[:, :, ::2, ::2]

        def upsample(x):
            cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]).to(device)], dim=3)
            cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
            cc = cc.permute(0,1,3,2)
            cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2).to(device)], dim=3)
            cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
            x_up = cc.permute(0,1,3,2)
            return conv_gauss(x_up, 4*LapLoss.gauss_kernel(channels=x.shape[1]))

        def conv_gauss(img, kernel):
            img = torch.nn.functional.pad(img, (2, 2, 2, 2), mode='reflect')
            out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
            return out

        current = img
        pyr = []
        for level in range(max_levels):
            filtered = conv_gauss(current, kernel)
            down = downsample(filtered)
            up = upsample(down)
            diff = current-up
            pyr.append(diff)
            current = down
        return pyr

    def __init__(self, max_levels=5, channels=3):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.gauss_kernel = LapLoss.gauss_kernel(channels=channels)

    def forward(self, input, target):
        pyr_input  = LapLoss.laplacian_pyramid(
                img=input, kernel=self.gauss_kernel, max_levels=self.max_levels)
        pyr_target = LapLoss.laplacian_pyramid(
                img=target, kernel=self.gauss_kernel, max_levels=self.max_levels)
        return sum(torch.nn.functional.l1_loss(a, b) for a, b in zip(pyr_input, pyr_target))

laploss = LapLoss().to(device)
def generate_smpl_params(image):
    """
    Generate SMPL parameters from an image using SMPLer-X.
    """
    with torch.no_grad():
        smpl_output = smpler_x(image)
    
    return {
        'pose': smpl_output.pose,
        'shape': smpl_output.shape,
        'joints3d': smpl_output.joints,
        'camera': smpl_output.camera
    }

def perspective_projection(joints3d, camera):
    """
    Perform perspective projection of 3D joints.
    This is a placeholder function - you need to implement this.
    """
    # Placeholder implementation
    return joints3d[:, :2]

def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=1e-4):
    model.device()
    model.train()

    trainable_params = [param for name, param in model.flownet.named_parameters() if param.requires_grad]
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.L1Loss()

    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (data, gt) in enumerate(train_loader):
            data, gt = data.to(device), gt.to(device)
            img0, img1 = data[:, :3], data[:, 3:]

            optimizer.zero_grad()

            # Forward pass
            pred = model.inference(img0, img1)

            # Compute RIFE loss
            rife_loss = criterion(pred, gt)

            # Generate SMPL parameters
            pred_smpl = generate_smpl_params(pred)
            gt_smpl = generate_smpl_params(gt)

            # Compute human loss
            human_loss_value = human_loss(pred_smpl, gt_smpl)

            # Compute census loss
            census_loss = census(pred, gt).mean()

            # Compute laplacian loss
            lap_loss = laploss(pred, gt).mean()

            # Combine losses
            total_loss = rife_loss + 0.1 * human_loss_value + 0.1*census_loss + 0.1*lap_loss  # Adjust the weight as needed

            epoch_loss += total_loss.item()

            # Backward pass and optimization
            total_loss.backward()
            optimizer.step()

            print(f"Batch {batch_idx + 1}/{len(train_loader)} - Total Loss: {total_loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # Validate the model
        validate_model(model, val_loader, device)

    print("Training complete.")

def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    criterion = nn.L1Loss()

    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(val_loader):
            data, gt = data.to(device), gt.to(device)
            img0, img1 = data[:, :3], data[:, 3:]

            pred = model.inference(img0, img1)

            # Compute RIFE loss
            rife_loss = criterion(pred, gt)

            # Generate SMPL parameters
            pred_smpl = generate_smpl_params(pred)
            gt_smpl = generate_smpl_params(gt)

            # Compute human loss
            human_loss_value = human_loss(pred_smpl, gt_smpl)

            # Combine losses
            total_loss = rife_loss + 0.1 * human_loss_value  # Adjust the weight as needed

            val_loss += total_loss.item()

            print(f"Validation Batch {batch_idx + 1}/{len(val_loader)} - Total Loss: {total_loss.item():.4f}")

    avg_val_loss = val_loss / len(val_loader)
    print(f"Validation completed. Average Loss: {avg_val_loss:.4f}")
    model.train()

# Main script
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    model = Model()
    model.load_model('train_log', -1)
    freeze_layers(model)

    # Load dataset
    full_dataset = TripletDataset('./ValidationTriplets', start_clip=6235, end_clip=6285)

    # Split dataset
    train_dataset, val_dataset = split_dataset(full_dataset, train_ratio=0.8)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    # Train the model
    train_model(model, train_loader, val_loader, device, epochs=1, learning_rate=1e-4)

    save_path = f"/model1.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved at {save_path}")
