import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import AutoTokenizer
import pandas as pd
import os
import numpy as np



class LLaVAComplaintDataset(Dataset):
    def __init__(self, df, image_root, processor, size):
        self.data = df.reset_index(drop=True)
        self.image_root = image_root
        #self.tokenizer = tokenizer
        self.processor = processor
        self.size = 224

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.image_root, row["image_path"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
    
        # Process image
        try:
            image = Image.open(image_path).convert("RGB")
            image = image.resize((224, 224))
        except Exception:
            print(f"[Skipping] Bad image at index {idx}: {image_path}")
            return self.__getitem__((idx + 1) % len(self))  # Retry with next index

        
        # Multimodal prompt (or plain)
        prompt = self.processor.apply_chat_template(
            [
                {"role": "user", "content": [{"type": "text", "text": row["text"]}, {"type": "image"}]}
            ],
            add_generation_prompt=True
        )
        
        encoding = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=False
        )


        label = int(row["label"])
        pixel_values = encoding["pixel_values"].squeeze(0)  # (3, H, W)

        assert pixel_values is not None, f"pixel_values is None for index {idx}"
        assert isinstance(pixel_values, torch.Tensor), f"pixel_values is not a tensor: {type(pixel_values)}"
        if pixel_values is None:
            print(f"[Warning] Failed image at index {idx}: {image_path}")
            print(f"[Skipping] pixel_values is None at index {idx}")
            return self.__getitem__((idx + 1) % len(self)) 

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "image_sizes": torch.tensor(image.size[::-1]),  # (H, W)
            "labels": encoding["input_ids"].squeeze(0),
        }