import os
import glob
import random
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class SyntheticSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8, image_size=(128, 128), use_augmentation=False):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.use_augmentation = use_augmentation
        self.samples = []

        self.base_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self._index_sequences()

    def _index_sequences(self):
        camera_folders = sorted(glob.glob(os.path.join(self.root_dir, "camera_*")))
        T = self.sequence_length
        required_len = T + 1

        for camera_path in camera_folders:
            sequence_folders = sorted(glob.glob(os.path.join(camera_path, "sequence_*")))
            for seq_path in sequence_folders:
                frame_paths = sorted(glob.glob(os.path.join(seq_path, "*.png")))
                num_frames = len(frame_paths)
                if num_frames >= required_len:
                    for i in range(num_frames - required_len + 1):
                        self.samples.append(frame_paths[i:i + required_len])

    def dilate_mask_tensor(self, tensor_img, kernel_size=3):
        tensor_img = tensor_img.unsqueeze(0)
        dilated = F.max_pool2d(tensor_img, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        return dilated.squeeze(0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths = self.samples[idx]
        clip = []
        clip_length = self.sequence_length + 1
        blink_start, blink_len, blink_mode = None, 0, None

        if self.use_augmentation and random.random() < 0.3:
            blink_start = random.randint(0, clip_length - 1)
            blink_len = random.randint(1, 3)
            blink_mode = random.choice(["zero", "dilate", "hole", "mixed"])

        for t, path in enumerate(frame_paths):
            img = Image.open(path).convert("L")
            img = self.base_transform(img)

            if self.use_augmentation:
                if blink_start is not None and blink_start <= t < blink_start + blink_len:
                    if blink_mode == "zero":
                        img = torch.zeros_like(img)
                    elif blink_mode == "dilate":
                        img = self.dilate_mask_tensor(img, kernel_size=random.choice([3, 5]))
                    elif blink_mode == "hole":
                        eraser = transforms.RandomErasing(p=1.0, scale=(0.1, 0.3), ratio=(0.3, 3.3), value=0)
                        img = eraser(img)
                    elif blink_mode == "mixed":
                        img = self.dilate_mask_tensor(img, kernel_size=random.choice([3, 5]))
                        if random.random() < 0.7:
                            eraser = transforms.RandomErasing(p=1.0, scale=(0.05, 0.2), ratio=(0.3, 3.3), value=0)
                            img = eraser(img)
                else:
                    img_pil = transforms.ToPILImage()(img)
                    if random.random() < 0.5:
                        img_pil = transforms.GaussianBlur(kernel_size=5)(img_pil)
                    if random.random() < 0.5:
                        img_pil = transforms.RandomAffine(degrees=10, scale=(0.9, 1.1), translate=(0.1, 0.1))(img_pil)
                    img = transforms.ToTensor()(img_pil)
                    if random.random() < 0.5:
                        eraser = transforms.RandomErasing(p=1.0, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=0)
                        img = eraser(img)
                    if random.random() < 0.3:
                        img = self.dilate_mask_tensor(img, kernel_size=random.choice([3, 5]))

            clip.append(img)

        clip = torch.stack(clip).float()
        T = self.sequence_length
        return {
            "input_seq": clip[:T],
            "target_frame": clip[T]
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dataset = SyntheticSequenceDataset(
        root_dir=r"synth/data/to/load",
        sequence_length=8,
        image_size=(128, 128),
        use_augmentation=True
    )

    sample = dataset[random.randint(0, len(dataset)-1)]
    frame_idx = random.randint(0, 7)
    img = sample["input_seq"][frame_idx].squeeze().numpy()

    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.show()
