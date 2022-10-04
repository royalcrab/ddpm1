# ddpm1

```first.ipynb
!pip install huggingface_hub
!huggingface-cli login
!git config --global credential.helper store
!pip install transformers scipy ftfy
!git clone https://github.com/huggingface/diffusers.git
!pip install git+https://github.com/huggingface/diffusers.git
```

## next

```next.py
import torch
from torch import autocast
import requests
from PIL import Image
from io import BytesIO
import os
from google.colab import drive

from diffusers import StableDiffusionImg2ImgPipeline

drive.mount('/content/drive')
os.chdir('/content/drive/MyDrive/colab')

# load the pipeline
device = "cuda"
pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)
```

# generate

```generate.py

outdir = "/content/drive/MyDrive/colab/sd1"
count = 10
prompt = "A glass of coffee float is on the table"

shitae = "/content/drive/MyDrive/colab/1.png"

init_image = Image.open(shitae).convert("RGB")
init_image = init_image.resize((768, 512))

if not os.path.exists(outdir):
  os.mkdir(outdir)
os.chdir(outdir)

for i in range(count):
  with autocast("cuda"):
    images = pipe(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)["sample"]
#    images = pipe(prompt=prompt, strength=0.75, guidance_scale=7.5)["sample"]

    images[0].save("image" + str(i) + ".png")
```
