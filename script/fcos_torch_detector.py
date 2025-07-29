import cv2
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor


# COCO class labels (index 0 is reserved and unused in COCO detection models)
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Load image using OpenCV and convert to RGB
image_path = 'fcos_torch_backend/script/image_000.png'
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Convert image to tensor and normalize
image_tensor = to_tensor(image_rgb).unsqueeze(0)  # Shape: [1, 3, H, W]

# Load the FCOS model with ResNet50-FPN backbone
model = fcos_resnet50_fpn(weights=FCOS_ResNet50_FPN_Weights.DEFAULT)
model.eval()

# Run the model
with torch.no_grad():
    outputs = model(image_tensor)[0]

# Draw predictions with confidence > 0.5
boxes = outputs['boxes']
scores = outputs['scores']
labels = outputs['labels']
confidence_threshold = 0.5

for box, score, label in zip(boxes, scores, labels):
    if score >= confidence_threshold:
        # Draw bounding box
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label and score
        class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        label_text = f'{class_name}: {score:.2f}'
        cv2.putText(image_bgr, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

# Show result using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FCOS Object Detection')
plt.show()
