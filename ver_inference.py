from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

#device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Load your model and processor
model_name_or_path = "result2/"  # or your HF model name
processor = AutoProcessor.from_pretrained(model_name_or_path)
model = LlavaOnevisionForConditionalGeneration.from_pretrained(model_name_or_path)
#pipe = pipeline("text-generation", model=model, tokenizer=processor.tokenizer, device_map="auto", token=hf_token)
model.to("cpu")
model.eval()

prompt = (
     "You are an expert review rewriter. "
    "First, determine whether the review is positive or negative. "
    "Then, rewrite the review to be much longer, more detailed, clearer, and more informative, elaborating on the detected sentiment. "
    "If the review is positive, expand on the positive aspects and provide more helpful context. "
    "If the review is negative, elaborate on the complaints and issues. "
    "Make the rewritten review as comprehensive as possible. Answer in plain text. Just the review and do not add any heading."
)

text = "The purple one bled even after I put it in oxy for several hours. What you see here is the laundry bag I had it in. I spilt water in the laundry bin and this is the result. I’m just glad the only things it bled onto were those sleeves.<br />Also, these are made for someone with a big head."
#text = prompt.format(review=text)

text2 = "5 different bottles had loose lids which caused the polish to pour out all throughout the box. I was so excited to get these colors and now that's just ruined with a sticky mess I honestly just don't even want to deal with having to clean up. I'd honestly like a replacement but it's not even worth the hassle of having to go out to do a return and ship it back 😔."
#text2 = prompt.format(review=text2)


messages = [
    {
        "role": "system",
        "content":  [
            {"type": "text", "text": prompt},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/fariha/custom/images/Theybleed.jpg"},
            {"type": "text", "text": f"Original review: {text}"},
        ],
    },
]

# Prepare multimodal prompt (example)
messages2 = [
    {
        "role": "system",
        "content":  [
            {"type": "text", "text": prompt},
        ],
    },
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/home/fariha/custom/images/Atotalmess.jpg"},
            {"type": "text", "text": f"Original review: {text2}"},
        ],
    },
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    max_length=2054,         # or whatever you need
    truncation=False         # <--- add this!
)
output = model.generate(**inputs, max_new_tokens=2054)


print(processor.decode(output[0][2:], skip_special_tokens=True))

inputs = processor.apply_chat_template(
    messages2,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
    max_length=2054,         # or whatever you need
    truncation=False         # <--- add this!
)
output = model.generate(**inputs, max_new_tokens=2054)


print(processor.decode(output[0][2:], skip_special_tokens=True))