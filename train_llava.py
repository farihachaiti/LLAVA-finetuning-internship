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

device = "cuda:0" if torch.cuda.is_available() else "cpu"
client = OpenAI(api_key='')

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)
# Choose a model (adjust as needed)
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # or any other supported model

# Load model and tokenizer (do this once, outside the function)
hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
hf_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quantization_config, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32).to(device)
hf_pipe = pipeline("text-generation", model=hf_model, tokenizer=hf_tokenizer)

def improve_review(review):
    prompt = f"Original review: {review}\n\nRewrite this review to be clearer, more informative, and more entailed."
    result = hf_pipe(review, max_new_tokens=1024, temperature=0.7, do_sample=True)
    return result[0]['generated_text'].split('\n', 1)[-1].strip()

# Apply to your DataFrame (this will be slow and cost money for large datasets!)


# Use GPU if available
device = "cuda:0" if torch.cuda.is_available() else "cpu"

torch.cuda.empty_cache()

# Save to CSV
#df.to_csv("complaints_with_images.csv", index=False, encoding="utf-8")
#df = pd.DataFrame(data_rows)
df = pd.read_csv("complaints_with_images.csv")[:50]
df = df.dropna(subset=['image_url', 'review_body', 'annotation'])
label_map = {"Complaint": 1, "Non-Complaint": 0}
df["label"] = df["annotation"].map(label_map)
def clean_text(text):
    return str(text).strip().lower()

df["text"] = df["review_body"].apply(lambda x: f"Review: {x}\nRewrite this review to be clearer, more informative, and more entailed.")
df["regenerated_review"] = df["text"].apply(improve_review)
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
model  = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf", quantization_config=quantization_config, torch_dtype="auto", low_cpu_mem_usage=True)
model.gradient_checkpointing_enable()
size = model.config.vision_config.image_size
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # adjust for your model
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

model.to(device)

tokenizer = AutoTokenizer.from_pretrained("llava-hf/llava-onevision-qwen2-0.5b-ov-hf")

train_dataset = LLaVAComplaintDataset(
    df=train_df,
    image_root="images/",
    processor=processor,
    size=size,
)

eval_dataset = LLaVAComplaintDataset(
    df=validation_df,
    image_root="images/",
    processor=processor,
    size=size,
)



test_dataset = LLaVAComplaintDataset(
    df=test_df,
    image_root="images/",
    processor=processor,
    size=size,
)

output_dir="result/"
# define training args
training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True if get_last_checkpoint(output_dir) is not None else False,
    num_train_epochs=3,
    warmup_steps=2,
    fp16=False,
    save_total_limit=2,
    logging_dir="logs/",
    learning_rate=float(2e-5),
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    disable_tqdm=True,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,
)

metric = load_metric("accuracy")
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)

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

# train model
trainer.train()

eval_result = trainer.evaluate(eval_dataset=test_dataset)

# writes eval result to file which can be accessed later in s3 ouput
with open(os.path.join("result/", "eval_results.txt"), "w") as writer:
    print(f"***** Eval results *****")
    for key, value in sorted(eval_result.items()):
        writer.write(f"{key} = {value}\n")
        print(f"{key} = {value}\n")

# Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
trainer.save_model(output_dir)
processor.save_pretrained(output_dir)

trainable = [n for n, p in model.named_parameters() if p.requires_grad]
print("Trainable parameters:", trainable)
