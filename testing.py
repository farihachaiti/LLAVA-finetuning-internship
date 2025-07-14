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

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from sklearn.model_selection import train_test_split
from evaluate import load as load_metric
# Use GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load the full split of the dataset
dataset = load_dataset(
    "McAuley-Lab/Amazon-Reviews-2023",
    "raw_review_All_Beauty",
    trust_remote_code=True,
    split="full"
)

# Prepare a list to store filtered data
data_rows = []

# Iterate through the dataset
for example in dataset:
    images = example.get("images", [])
    has_image = isinstance(images, list) and len(images) > 0
    rating = example.get("rating", 0)
    is_complaint = rating <= 3

    if has_image:
        annotation = "Complaint" if is_complaint else "Non-Complaint"
        image_url = images[0].get("large_image_url", "")
        #print(image_url)
        data_rows.append({
            "review_id": example.get("title", "None"),
            "image_url": image_url,
            "rating": rating,
            "annotation": annotation,
            "review_body": example.get("text", "None")
        })

# Create a pandas DataFrame
df = pd.DataFrame(data_rows)

# Save to CSV
df.to_csv("complaints_with_images.csv", index=False, encoding="utf-8")

df2 = pd.read_csv("your_file.csv")
df2 = df.dropna(subset=['image_url', 'review_body', 'annotation'])
label_map = {"Complaint": 1, "Non-Complaint": 0}
df2["label"] = df["annotation"].map(label_map)
def clean_text(text):
    return str(text).strip().lower()

df2["text"] = df2["review_body"].apply(clean_text)
os.makedirs("images", exist_ok=True)

def download_image(url, review_id):
    try:
        response = requests.get(url, timeout=5)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        path = f"images/{review_id}.jpg"
        img.save(path)
        return path
    except Exception as e:
        print(f"Failed for {url}: {e}")
        return None

df2["image_path"] = df2.apply(lambda row: download_image(row["image_url"], row["review_id"]), axis=1)
df2 = df2.dropna(subset=["image_path"])
train_df, test_df = train_test_split(df2, test_size=0.2, stratify=df2["label"], random_state=42)
#train2_df, validation_df = train_test_split(train_df, test_size=0.1, stratify=train2_df["label"], random_state=42)
# Preview the data
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, torch_dtype="auto", device_map="auto")
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
          min_pixels=224*224,  # Standard ViT patch size
          max_pixels=336*336)


tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
output_dir="finetune/"
# define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True if get_last_checkpoint(output_dir) is not None else False,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    warmup_steps=2,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    logging_dir=f"{output_dir}/logs",
    learning_rate=float(2e-5),
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    disable_tqdm=True,
)

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)
# create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_df,
    eval_dataset=test_df,
    tokenizer=tokenizer,
)


# train model
trainer.train()


def improve_review(review):
    completion = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert review rewriter. "
                    "First, determine whether the review is positive or negative. "
                    "Then, rewrite the review to be much longer, more detailed, clearer, and more informative, elaborating on the detected sentiment. "
                    "If the review is positive, expand on the positive aspects and provide more helpful context. "
                    "If the review is negative, elaborate on the complaints and issues. "
                    "Make the rewritten review as comprehensive as possible."
                ),
            },
            {
                "role": "user",
                "content": f"Original review: {review}",
            }
        ],
    )
    return completion.choices[0].message.content