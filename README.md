# ddpm1

```first.ipynb
!pip install huggingface_hub
!huggingface-cli login
!git config --global credential.helper store
!pip install transformers scipy ftfy
!git clone https://github.com/huggingface/diffusers.git
!pip install git+https://github.com/huggingface/diffusers.git
```

https://huggingface.co/ でアカウントを作成し、トークンを生成する。
アカウントを作るとメールが送られてくるので、メールにあるリンク先を開いて、アカウントを有効化する。

## next

Google Drive のルートディレクトリ（トップのフォルダ）に "colab" という名前のフォルダを作成しておく。

```next.py
import torch
from torch import autocast
import requests
from PIL import Image
from io import BytesIO
import os
from google.colab import drive

from diffusers import StableDiffusionImg2ImgPipeline
from diffusers import StableDiffusionPipeline

drive.mount('/content/drive')

codir = "/content/drive/MyDrive/colab/"
if not os.path.exists(codir):
  os.mkdir(codir)

# load the pipeline
device = "cuda"
pipe1 = StableDiffusionImg2ImgPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="fp16", 
    torch_dtype=torch.float16,
    use_auth_token=True
).to(device)

pipe2 = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", 
	use_auth_token=True
).to("cuda")
```

一回目の実行時には、URL のついたエラーが表示される。
表示された URL を開くと huggingface.co のサイトへ飛ぶので、そのページに表示されているライセンスに同意する。
同意した状態で、再度上のコードを実行すると、実行できるはず。

# generate (下絵なし)

```generate.py
outdir = "/content/drive/MyDrive/colab/sd1"
count = 10
prompt = "A glass of coffee float is put on he table"

if not os.path.exists(outdir):
  os.mkdir(outdir)
os.chdir(outdir)

for i in range(count):
    with autocast("cuda"):
        images = pipe1(prompt)["sample"]  
        images[0].save("image" + str(i) + ".png")
```

# generate (下絵あり)

Google Drive の colab フォルダ以下に、1.png というファイルを置いておく（下絵として）。

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
    images = pipe2(prompt=prompt, init_image=init_image, strength=0.75, guidance_scale=7.5)["sample"]

    images[0].save("image" + str(i) + ".png")
```
