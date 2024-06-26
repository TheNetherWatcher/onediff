from diffusers import AutoPipelineForText2Image
import torch
import time

pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
).to("cuda")

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"
t = time.time()
image = pipeline_text2image(prompt=prompt).images[0]
print(time.time() - t)
print(image.size)
image
