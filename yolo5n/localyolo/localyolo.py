import torch

model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt")  # PyTorch

# Images
imgs = [f'./data/people({i+1}).jpg' for i in range(10)]  # batch of images
# Inference
results = model(imgs)

# Results
results.print()
results.save()  # or .show()

results.xyxy[0]  # img1 predictions (tensor)
results.pandas().xyxy[0]  # img1 predictions (pandas)
