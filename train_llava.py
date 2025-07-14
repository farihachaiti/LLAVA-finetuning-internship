import torch
import pandas as pd
from datasets import load_dataset
import os
import requests
from PIL import Image
from io import BytesIO
import argparse
import logging
import os
import random
import sys
from datasetComplaint import LLaVAComplaintDataset
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer, LlavaOnevisionForConditionalGeneration
import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from sklearn.model_selection import train_test_split
from evaluate import load as load_metric
from peft import LoraConfig, get_peft_model, TaskType
from transformers import DefaultDataCollator
from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import TrainerCallback
import json
import matplotlib.pyplot as plt
from datetime import datetime
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
load_dotenv()

# Set CUDA memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
torch.cuda.empty_cache()
device = "cuda:0" if torch.cuda.is_available() else "cpu"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 instead of bfloat16
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,  # Enable double quantization
)

df = pd.read_csv("complaints_with_regenerated_reviews.csv")
df = df.dropna(subset=['image_url', 'review_body', 'annotation', 'regenerated_review'])

# Clean annotation values to avoid NaN in label mapping
df['annotation'] = df['annotation'].str.strip().str.title()

label_map = {"Complaint": 1, "Non-Complaint": 0}
df["label"] = df["annotation"].map(label_map)

# Drop rows with NaN labels (invalid annotation values)
df = df.dropna(subset=['label'])

def clean_text(text):
    return str(text).strip().lower()
df["text"] = df["review_body"].apply(clean_text)
df["regenerated_review"] = df["regenerated_review"].apply(clean_text)

#df["regenerated_review"] = df["text"].apply(improve_review)
os.makedirs("images", exist_ok=True)
os.makedirs("result", exist_ok=True)
def download_image(url, review_id):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        filename = f"{review_id.replace(' ', '')[:20]}.jpg"
        path = os.path.join("images", filename)
        img.save(path)
        return filename
    except Exception as e:
        print(f"Failed for {url}: {e}")
        return None
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except Exception:
        return False


# Step 1: Download images
df["image_path"] = df.apply(lambda row: download_image(row["image_url"], row["review_id"]), axis=1)

# Step 2: Drop rows where download failed
df = df[df["image_path"].notna()]

# Step 3: Keep only rows with valid image files
def safe_is_valid_image(p):
    if p is None:
        return False
    full_path = os.path.join("images", p)
    return is_valid_image(full_path)

df = df[df["image_path"].apply(safe_is_valid_image)].reset_index(drop=True)

def get_image_size(image_path):
    try:
        with Image.open(os.path.join("images", image_path)) as img:
            return img.size[::-1]  # (height, width)
    except Exception:
        return None


df["image_sizes"] = df["image_path"].apply(get_image_size)
df = df[df["image_sizes"].notna()].reset_index(drop=True)

train_df, validation_df = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=42)
train2_df, test_df = train_test_split(train_df, test_size=0.1, stratify=train_df["label"], random_state=42)
# Preview the data
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    max_length=128  # or 256
)
model  = LlavaOnevisionForConditionalGeneration.from_pretrained(
    "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
    quantization_config=quantization_config,
    torch_dtype="auto",
    low_cpu_mem_usage=True,
    device_map="auto",  # <--- add this
)
model.config.use_cache = False  # Add this after model loading
model.gradient_checkpointing_enable()
# Reduce LoRA parameters to save memory
lora_config = LoraConfig(
    r=4,  # Reduce from 8 to 4
    lora_alpha=16,  # Reduce from 32 to 16
    target_modules=["q_proj", "v_proj"],  # adjust for your model
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

model.to(device)
torch.cuda.empty_cache()
tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", use_fast=True)

train_dataset = LLaVAComplaintDataset(
    df=train_df,
    image_root="images/",
    processor=processor,
    size=224,
)

eval_dataset = LLaVAComplaintDataset(
    df=validation_df,
    image_root="images/",
    processor=processor,
    size=224,
)



test_dataset = LLaVAComplaintDataset(
    df=test_df,
    image_root="images/",
    processor=processor,
    size=224,
)

output_dir="result2/"
torch.cuda.empty_cache()
# define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True if get_last_checkpoint(output_dir) is not None else False,
    num_train_epochs=1,  # Reduce from 3 to 1
    warmup_steps=1,  # Reduce from 2 to 1
    fp16=False,  # Disable mixed precision if you're using 4-bit quantization
    bf16=torch.cuda.is_bf16_supported(),  # Try bfloat16 if available
    optim="paged_adamw_8bit",  # Use 8-bit optimizer
    gradient_checkpointing=True,
    save_total_limit=1,  # Reduce from 2 to 1
    logging_dir="logs/",
    learning_rate=float(1e-5),  # Reduce from 2e-5 to 1e-5
    weight_decay=0.01,
    # metric_for_best_model="accuracy", # Removed as per instructions
    disable_tqdm=False,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,  # Reduce from 2 to 1
    eval_accumulation_steps=1,  # Avoids large logit buffers
    torch_empty_cache_steps=1,  # Periodically clear cache
    max_grad_norm=0.5,  # Add gradient clipping
    dataloader_pin_memory=False,  # Disable to save memory
    remove_unused_columns=False,
    ddp_find_unused_parameters=False,  # Disable unused parameter detection
    dataloader_num_workers=0,  # Disable multiprocessing
)

# Remove accuracy metric loading and computation
# Remove complaint ratio and classification metrics
# Only keep loss reporting and (optionally) sample output printing

# Remove metric = load_metric("accuracy") and compute_metrics
# Remove metric_for_best_model="accuracy" from TrainingArguments if present

# For most Hugging Face models, this works:
data_collator = DefaultDataCollator()

# For multi-modal, you may need:
def multimodal_data_collator(features):
    pixel_values = torch.stack([f["pixel_values"] for f in features])  # stack, not list
    labels = torch.stack([f["labels"] for f in features])
    image_sizes = torch.stack([f["image_sizes"] for f in features])
    batch = tokenizer.pad(
        [{k: v for k, v in f.items() if k not in ["pixel_values", "labels", "image_sizes"]} for f in features],
        padding=True,
        return_tensors="pt"
    )
    batch["pixel_values"] = pixel_values
    batch["image_sizes"] = image_sizes
    batch["labels"] = labels
    return batch
torch.cuda.empty_cache()
# create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=multimodal_data_collator,
)

def compute_loss(model, inputs, return_outputs=False, **kwargs):
    print("Model training mode:", model.training)
    outputs = model(**inputs)
    loss = outputs.loss
    loss.requires_grad = True
    print("Loss requires_grad:", loss.requires_grad)
    return (loss, outputs) if return_outputs else loss

trainer.compute_loss = compute_loss
# Add this callback to your Trainer
class MemoryCleanupCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        torch.cuda.empty_cache()

trainer.add_callback(MemoryCleanupCallback())
torch.cuda.empty_cache()
# train model
trainer.train()

trainer.evaluate(eval_dataset=test_dataset)

# Generate comprehensive evaluation report

# Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print("Trainable parameters:", trainable)

# After evaluation, print or save a few sample outputs for qualitative review
sample_outputs_path = os.path.join(output_dir, "sample_rewrites.txt")
with open(sample_outputs_path, "w") as f:
    for i in range(5):  # Save 5 sample outputs
        sample = test_dataset[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        image_sizes = sample["image_sizes"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                max_new_tokens=128
            )
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        f.write(f"Sample {i+1} rewritten review:\n{generated}\n{'-'*40}\n")
        print(f"Sample {i+1} rewritten review:\n{generated}\n{'-'*40}")