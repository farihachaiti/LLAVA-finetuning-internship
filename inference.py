import torch
from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load your fine-tuned model & processor (replace with your checkpoint path if local)
model_name_or_path = "result/"  # or "llava-hf/llava-v1.6-mistral-7b-hf" if just base

processor = AutoProcessor.from_pretrained(model_name_or_path)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name_or_path)
model.to("cpu")
model.eval()



# Prepare multimodal prompt (example)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/fariha/custom/images/Theybleed.jpg"},
            {"type": "text", "text": "The purple one bled even after I put it in oxy for several hours. What you see here is the laundry bag I had it in. I spilt water in the laundry bin and this is the result. I’m just glad the only things it bled onto were those sleeves.<br />Also, these are made for someone with a big head."},
        ],
    },
]

# Prepare multimodal prompt (example)
messages2 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/fariha/custom/images/Atotalmess.jpg"},
            {"type": "text", "text": "5 different bottles had loose lids which caused the polish to pour out all throughout the box. I was so excited to get these colors and now that's just ruined with a sticky mess I honestly just don't even want to deal with having to clean up. I'd honestly like a replacement but it's not even worth the hassle of having to go out to do a return and ship it back 😔"},
        ],
    },
]
inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=500)


print(processor.decode(output[0][2:], skip_special_tokens=True))

inputs = processor.apply_chat_template(messages2, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=500)


print(processor.decode(output[0][2:], skip_special_tokens=True))