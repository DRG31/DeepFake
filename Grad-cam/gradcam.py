import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
from timm import create_model  # Using timm for pre-trained models

# ---------------------------
# Dataset Class
# ---------------------------
class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = 1 if "fake" in img_path.lower() else 0  # Label as 1 (fake) or 0 (real)
        if self.transform:
            image = self.transform(image)
        return image, label, img_path

# ---------------------------
# Grad-CAM Implementation
# ---------------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        layer = dict([*self.model.named_modules()])[self.target_layer]
        layer.register_forward_hook(forward_hook)
        layer.register_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_score = output[:, class_idx]
        class_score.backward()

        gradients = self.gradients.cpu().detach().numpy()
        activations = self.activations.cpu().detach().numpy()

        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling
        cam = np.zeros(activations.shape[2:], dtype=np.float32)

        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]

        cam = np.maximum(cam, 0)  # ReLU
        cam = cam / cam.max()  # Normalize
        return cam

# ---------------------------
# Utility Functions
# ---------------------------
def overlay_cam_on_image(image, cam, alpha=0.5):
    cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
    cam_colormap = plt.cm.jet(np.array(cam_resized) / 255.0)[:, :, :3]  # Apply colormap
    cam_colormap = (cam_colormap * 255).astype(np.uint8)

    blended = Image.blend(image, Image.fromarray(cam_colormap), alpha=alpha)
    return blended

def save_comparison(image, cam, overlay, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original Image
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # CAM
    axes[1].imshow(cam, cmap="jet")
    axes[1].set_title("CAM")
    axes[1].axis("off")

    # Overlay
    axes[2].imshow(overlay)
    axes[2].set_title("Overlay")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------------------------
# Main Code
# ---------------------------
if __name__ == "__main__":
    # Load the pretrained model
    model_path = "T:/Study Material/UQ/REIT4842-Thesis/DeepfakeBench/training/pretrained/xception-b5690688.pth"  # Replace with your .pth file

    # Load XceptionNet model using timm
    model = create_model("xception", pretrained=False, num_classes=2)  # Binary classification
    checkpoint = torch.load(model_path)

    for key in checkpoint.keys():
        if "pointwise.weight" in key and checkpoint[key].ndimension() == 2:
            checkpoint[key] = checkpoint[key].unsqueeze(-1).unsqueeze(-1)

    model.load_state_dict(checkpoint)
    model.eval()

    # Set the target layer for Grad-CAM (use the name of the last convolutional layer)
    target_layer = "block12.rep.6"
    cam_extractor = GradCAM(model, target_layer)

    # Define preprocessing
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Xception requires 299x299 input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # Dataset paths (update these with real paths)
    real_images_path = "T:/Study Material/UQ/REIT4842-Thesis/DeepfakeBench/datasets/rgb/FaceForensics++/original_sequences/actors/c23/frames"  # Directory containing real images
    fake_images_path = "T:/Study Material/UQ/REIT4842-Thesis/DeepfakeBench/datasets/rgb/FaceForensics++/manipulated_sequences/Deepfakes/c23/frames"  # Directory containing fake images

    real_images = [os.path.join(real_images_path, img) for img in os.listdir(real_images_path)]
    fake_images = [os.path.join(fake_images_path, img) for img in os.listdir(fake_images_path)]

    # Combine datasets
    all_images = real_images + fake_images
    dataset = ImageDataset(all_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Output directory
    output_dir = "gradcam_results"
    os.makedirs(output_dir, exist_ok=True)

    # Process each image
    for input_tensor, label, img_path in dataloader:
        input_tensor = input_tensor.cuda() if torch.cuda.is_available() else input_tensor
        model = model.cuda() if torch.cuda.is_available() else model

        # Forward pass and Grad-CAM generation
        output = model(input_tensor)
        class_idx = output.argmax().item()
        cam = cam_extractor.generate(input_tensor, class_idx)

        # Convert and overlay
        image = Image.open(img_path[0]).convert("RGB")
        cam_resized = Image.fromarray((cam * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR)
        overlay = overlay_cam_on_image(image, cam)

        # Save comparison
        save_path = os.path.join(output_dir, f"{os.path.basename(img_path[0])}_comparison.png")
        save_comparison(image, cam_resized, overlay, save_path)

        print(f"Saved: {save_path}")
