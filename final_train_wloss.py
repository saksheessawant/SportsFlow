import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
from train_log.RIFE_HDv3 import Model
import torch.nn.functional as F
from smplx import SMPL 
from facenet_pytorch import MTCNN 
x
# Face Detection Module using a Pre-trained Model
class FaceDetectionModule:
    def __init__(self, device='cuda'):
        self.device = device
        self.detector = MTCNN(keep_all=True, device=device)

    def forward(self, img):
        batch_keypoints = []
        for i in range(img.size(0)):
            frame = img[i].permute(1, 2, 0).detach().cpu().numpy()  # Convert CHW -> HWC and detach gradients
            boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)
            if landmarks is not None:
                batch_keypoints.append(torch.tensor(landmarks[0]))  # Use the first detected face
            else:
                batch_keypoints.append(torch.zeros((68, 2)))  # Default zero keypoints if detection fails
        return torch.stack(batch_keypoints).to(img.device)

# SMPL Model for 3D Human Mesh Reconstruction
class SMPLModel:
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.smpl = SMPL(model_path, batch_size=1).to(device)

    def forward(self, img):
        batch_size = img.size(0)
        # Example: Generate body parameters (replace with extracted parameters from input images if available)
        betas = torch.zeros(batch_size, 10).to(self.device)  # Shape parameters
        body_pose = torch.zeros(batch_size, 69).to(self.device)  # Body pose
        global_orient = torch.zeros(batch_size, 3).to(self.device)  # Global orientation
        transl = torch.zeros(batch_size, 3).to(self.device)  # Translation

        # Compute the SMPL vertices
        output = self.smpl(betas=betas, body_pose=body_pose, global_orient=global_orient, transl=transl)
        return output.vertices  # Return 3D vertices

# Laplacian Pyramid Function with Padding (Reduced Levels)
def laplacian_pyramid(img, max_levels=2):
    current = img
    pyramid = []
    h, w = current.size(-2), current.size(-1)
    pad_h, pad_w = (h % 2) != 0, (w % 2) != 0
    if pad_h or pad_w:
        current = F.pad(current, (0, pad_w, 0, pad_h), mode='reflect')

    for _ in range(max_levels):
        down = F.avg_pool2d(current, kernel_size=2, stride=2)
        up = F.interpolate(down, size=current.size()[-2:], mode='bilinear', align_corners=False)
        pyramid.append(current - up)
        current = down

    return pyramid

# Triplet Dataset
class TripletDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.samples = sorted(os.listdir(self.root_dir))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        triplet_dir = os.path.join(self.root_dir, self.samples[idx])
        img0 = ToTensor()(cv2.imread(os.path.join(triplet_dir, "img0.jpg")))
        img1 = ToTensor()(cv2.imread(os.path.join(triplet_dir, "img1.jpg")))
        gt = ToTensor()(cv2.imread(os.path.join(triplet_dir, "gt.jpg")))
        return torch.cat((img0, img1), dim=0), gt

# Loss functions
def human_aware_loss(pred, gt, face_detector, smpl_model):
    l_basic = nn.L1Loss()(pred, gt)
    pred_kpts, gt_kpts = face_detector.forward(pred), face_detector.forward(gt)
    l_facekpt = nn.MSELoss()(pred_kpts, gt_kpts)
    pred_smpl, gt_smpl = smpl_model.forward(pred), smpl_model.forward(gt)
    l_human3d = nn.MSELoss()(pred_smpl, gt_smpl)
    pred_lap, gt_lap = laplacian_pyramid(pred), laplacian_pyramid(gt)
    l_lap = sum(nn.L1Loss()(p, g) for p, g in zip(pred_lap, gt_lap))
    l_census = nn.L1Loss()(pred, gt)
    return l_basic + l_facekpt + l_human3d + l_census + l_lap


def train_model(model, train_loader, device, face_detector, smpl_model, epochs=1, log_interval=10):
    model.device()
    model.train()
    optimizer = optim.Adam(model.flownet.parameters(), lr=1e-4)

    for epoch in range(epochs):
        total_loss = 0
        print(f"Starting Epoch {epoch + 1}")
        for batch_idx, (data, gt) in enumerate(train_loader):
            data, gt = data.to(device), gt.to(device)
            img0, img1 = data[:, :3], data[:, 3:]

            optimizer.zero_grad()
            pred = model.inference(img0, img1)
            loss = human_aware_loss(pred, gt, face_detector, smpl_model)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if batch_idx % log_interval == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Log average loss for the epoch
        print(f"Epoch {epoch + 1} completed. Average Loss: {total_loss / len(train_loader):.4f}")

        # Save the model at the end of the epoch
        save_path = f"./finall_acv_model_epoch_{epoch + 1}.pth"
        torch.save(model.flownet.state_dict(), save_path)
        print(f"Model parameters saved at {save_path}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model and freeze layers
    model = Model()
    model.load_model('train_log', -1)
    print("Model loaded successfully")
    
    # Initialize modules
    face_detector = FaceDetectionModule(device=device)
    smpl_model = SMPLModel(model_path="./smpl_model", device=device)
    print("Face Detector and SMPL Model initialized")

    # Load dataset
    dataset = TripletDataset('./ValidationTriplets')
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    print("Dataset loaded successfully")

    # Train
    train_model(model, train_loader, device, face_detector, smpl_model, epochs=100, log_interval=1000)
