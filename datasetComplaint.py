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
        #prompt = f"<image> {row['text']}"
        
        target = row["regenerated_review"]
        
        prompt = (
            "You are an expert review rewriter. "
            "First, determine whether the review is positive or negative. "
            "Then, rewrite the review to be much longer, more detailed, clearer, and more informative, elaborating on the detected sentiment. "
            "If the review is positive, expand on the positive aspects and provide more helpful context. "
            "If the review is negative, elaborate on the complaints and issues. "
            "Make the rewritten review as comprehensive as possible. Answer in plain text. Just the review and do not add any heading."
        )
        messages = [
            {"role": "system", 
            "content": [
                {"type": "text", "text": prompt}
            ]},
            {"role": "user", 
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Original review: {row['text']}"}
            ]},
            {"role": "assistant", 
            "content": [
                {"type": "text", "text": target}
            ]}
        ]
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=False)
     

        # Tokenize prompt separately to get its length
        prompt_encoding = self.processor.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,      # or any smaller value you want
            truncation=True
        )
        prompt_length = prompt_encoding["input_ids"].shape[1]

        # Tokenize full input (prompt + target)
        encoding = self.processor(
            images=image,
            text=prompt,
            text_target=target,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2048,
        )
        input_ids = encoding["input_ids"].squeeze(0)
        labels = input_ids.clone()
        labels[:prompt_length] = -100  # mask out the prompt
        pixel_values = encoding["pixel_values"].squeeze(0)

        label = int(row["label"])

        assert pixel_values is not None, f"pixel_values is None for index {idx}"
        assert isinstance(pixel_values, torch.Tensor), f"pixel_values is not a tensor: {type(pixel_values)}"
        if pixel_values is None:
            print(f"[Warning] Failed image at index {idx}: {image_path}")
            print(f"[Skipping] pixel_values is None at index {idx}")
            return self.__getitem__((idx + 1) % len(self)) 

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,
            "image_sizes": encoding["image_sizes"].squeeze(0),
        }