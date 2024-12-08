import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.transforms import ToTensor
from train_log.RIFE_HDv3 import Model

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


# Training function
def train_model(model, train_loader, val_loader, device, epochs=10, learning_rate=1e-4):
    model.device()
    model.train()

    # Collect trainable parameters manually
    trainable_params = [param for name, param in model.flownet.named_parameters() if param.requires_grad]

    # Define optimizer and loss function
    optimizer = optim.Adam(trainable_params, lr=learning_rate)
    criterion = nn.L1Loss()

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0.0
        print(f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, (data, gt) in enumerate(train_loader):
            data, gt = data.to(device), gt.to(device)

            # Split concatenated inputs
            img0, img1 = data[:, :3], data[:, 3:]

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            pred = model.inference(img0, img1)

            # Compute loss
            loss = criterion(pred, gt)
            epoch_loss += loss.item()

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

        # Validate the model
        validate_model(model, val_loader, device)

    print("Training complete.")


# Validation function
def validate_model(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    criterion = nn.L1Loss()

    with torch.no_grad():
        for batch_idx, (data, gt) in enumerate(val_loader):
            data, gt = data.to(device), gt.to(device)

            # Split concatenated inputs
            img0, img1 = data[:, :3], data[:, 3:]

            # Perform inference
            pred = model.inference(img0, img1)

            # Compute loss
            loss = criterion(pred, gt)
            val_loss += loss.item()

            print(f"Validation Batch {batch_idx + 1}/{len(val_loader)} - Loss: {loss.item():.4f}")

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