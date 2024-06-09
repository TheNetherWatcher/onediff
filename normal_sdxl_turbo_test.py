from diffusers import AutoPipelineForText2Image
import torch
import time

pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cuda")

prompt = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."

t = time.time()
image = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
print(time.time() -t)
print(image.size)
image
