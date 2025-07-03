import requests
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import sagemaker
import boto3
from sagemaker.huggingface import HuggingFace
import HuggingFace.datasets as datasets

device = "cuda:0" if torch.cuda.is_available() else "cpu"

from datasets import load_dataset

dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", "raw_review_All_Beauty", trust_remote_code=True, split="train")
print(dataset[0])

'''torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", quantization_config=quantization_config, torch_dtype="auto", device_map="auto")

# default processer
min_pixels = 256 * 28 * 28  # Minimum token allocation
max_pixels = 512 * 28 * 28  # Reduced max token allocation
processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf",
          min_pixels=224*224,  # Standard ViT patch size
          max_pixels=336*336)


try:
	role = sagemaker.get_execution_role()
except ValueError:
	iam = boto3.client('iam')
	role = iam.get_role(RoleName='sagemaker_execution_role')['Role']['Arn']
		
hyperparameters = {
	'model_name_or_path':'llava-hf/llava-v1.6-mistral-7b-hf',
	'output_dir':'/opt/ml/model'
	# add your remaining hyperparameters
	# more info here https://github.com/huggingface/transformers/tree/v4.49.0/path/to/script
}

# git configuration to download our fine-tuning script
git_config = {'repo': 'https://github.com/huggingface/transformers.git','branch': 'v4.49.0'}

# creates Hugging Face estimator
huggingface_estimator = HuggingFace(
	entry_point='train.py',
	source_dir='./path/to/script',
	instance_type='ml.p3.2xlarge',
	instance_count=1,
	role=role,
	git_config=git_config,
	transformers_version='4.49.0',
	pytorch_version='2.5.1',
	py_version='py311',
	hyperparameters = hyperparameters
)

# starting the train job
huggingface_estimator.fit()