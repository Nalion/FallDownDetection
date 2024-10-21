import torch

model = torch.hub.load("Nalion/FallDownDetection", "custom", path="yolov5n.pt")  # PyTorch

# Images
imgs = [f'./data/people({i+1}).jpg' for i in range(10)]  # batch of images
# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
