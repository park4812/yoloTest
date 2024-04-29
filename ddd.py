import torch
import cv2
import matplotlib.pyplot as plt
from utils.torch_utils import select_device

def imShow(path):

  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image,(3*width, 3*height), interpolation = cv2.INTER_CUBIC)

  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis("off")
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()
"""
print("g")
print(torch.cuda.device_count())
print(torch.version.cuda)
print( torch.version )

print(torch.cuda.is_available())


print(torch.cuda.get_device_name(0))

"""
print("------------")
device = select_device('0')
print(device)

import torch
import torchvision

print("1")
boxes = torch.tensor([[25, 25, 200, 200], [50, 50, 150, 150]], dtype=torch.float32).cuda()
print("2")
scores = torch.tensor([0.9, 0.8], dtype=torch.float32).cuda()
print("3")

# CUDA 백엔드에서 nms 실행
keep = torchvision.ops.nms(boxes, scores, 0.5)
print(keep)