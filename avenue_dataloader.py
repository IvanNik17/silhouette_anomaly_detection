import os
import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class AvenueMaskSequenceDataset(Dataset):
    def __init__(self, root_dir, sequence_length=8, image_size=(128, 128)):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.samples = []

        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize(image_size),
            transforms.ToTensor()
        ])

        self._index_sequences()

    def _index_sequences(self):
        subfolders = sorted(glob.glob(os.path.join(self.root_dir, "*")))
        for subfolder in subfolders:
            frame_paths = sorted(glob.glob(os.path.join(subfolder, "*.jpg")))
            if len(frame_paths) >= self.sequence_length:
                for i in range(len(frame_paths) - self.sequence_length + 1):
                    self.samples.append(frame_paths[i:i + self.sequence_length])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        clip = [self.transform(Image.open(p).convert("L")) for p in self.samples[idx]]
        clip = torch.stack(clip)
        return {
            "input_seq": clip[:-1],
            "target_frame": clip[-1]
        }


if __name__ == "__main__":
    dataset = AvenueMaskSequenceDataset(
        "dir_of_masks",
        sequence_length=8
    )
    print(f"Found {len(dataset)} sequences.")
    sample = dataset[0]
    print("Input sequence shape:", sample["input_seq"].shape)
    print("Target frame shape:", sample["target_frame"].shape)