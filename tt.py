import torch

# run: python /data/coding/upload-data/data/aimo/tt.py

print(torch.cuda.is_available())
gpu_count = torch.cuda.device_count()
for i in range(gpu_count):
    props = torch.cuda.get_device_properties(i)
    print(f"GPU {i}：名称 = {props.name}, 显存大小 = {props.total_memory / (1024**3):.2f} GB")