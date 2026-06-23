from datasets import load_dataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class PathVQADataset(Dataset):
    def __init__(self, hf_dataset):
        """
        hf_dataset: HuggingFace dataset (already loaded split)
        """
        self.dataset = hf_dataset

        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # --- Image ---
        # HF dataset already provides PIL image
        raw_image = sample["image"].convert('RGB')
        img = self.transform(raw_image)

        # --- Question & Answer ---
        question = sample["question"]
        answer = sample["answer"]

        return img, question, answer


pathvqa_valid = load_dataset(
    "parquet",
    data_files={"validation": "hf://datasets/flaviagiammarino/path-vqa/data/validation-*.parquet"},
    split="validation"
)

subset = pathvqa_valid.shuffle(seed=42).select(range(3000))
split_dataset = subset.train_test_split(test_size=0.2, seed=42)

# split_dataset = pathvqa_valid.train_test_split(test_size=0.2, seed=42)

train_data = split_dataset["train"]
val_data = split_dataset["test"]
print(f'split: train = {len(train_data)} and test={len(val_data)}')


def preview_dataset(idx=3):
    plt.axis('off')
    print('samplesize:', len(pathvqa_valid))
    print(pathvqa_valid[idx].keys())
    print('image reso;ution:', np.array(pathvqa_valid[idx]['image']).shape)
    plt.imshow(pathvqa_valid[idx]['image'])
    plt.title(
        f"Q: {pathvqa_valid[idx]['question']}\nA: {pathvqa_valid[idx]['answer']}",
        fontsize=12
    )


def preview_train_sample(idx=1):
    train_dataset = PathVQADataset(train_data)
    img, question, answer = train_dataset[idx]
    print('image resolution:', img.size())
    plt.axis('OFF')
    plt.imshow(img.permute(1, 2, 0))
    plt.title(f'Q: {question}\nA: {answer}', fontsize=12)
