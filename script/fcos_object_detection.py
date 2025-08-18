import cv2
import matplotlib.pyplot as plt

import torch
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection import FCOS_ResNet50_FPN_Weights
from torchvision.transforms.functional import to_tensor


# Correct COCO class mapping using actual category IDs (with gaps)
COCO_INSTANCE_CATEGORY_NAMES = {
    0: '__background__',  # Background class (though typically not returned in detections)
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
    11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
    16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
    22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
    28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
    35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite',
    39: 'baseball bat', 40: 'baseball glove', 41: 'skateboard', 42: 'surfboard',
    43: 'tennis racket', 44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork',
    49: 'knife', 50: 'spoon', 51: 'bowl', 52: 'banana', 53: 'apple',
    54: 'sandwich', 55: 'orange', 56: 'broccoli', 57: 'carrot', 58: 'hot dog',
    59: 'pizza', 60: 'donut', 61: 'cake', 62: 'chair', 63: 'couch',
    64: 'potted plant', 65: 'bed', 67: 'dining table', 70: 'toilet', 72: 'tv',
    73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard', 77: 'cell phone',
    78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink', 82: 'refrigerator',
    84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors', 88: 'teddy bear',
    89: 'hair drier', 90: 'toothbrush'}
# IDs 12, 26, 29, 30, 45, 66, 68, 69, 71, 83 are not used
# they were categories in early drafts of COCO but removed.

# Load image using OpenCV and convert to RGB
image_path = './test/image_000.png'
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

detection_count = 0

print('Detected objects:')
for box, score, label in zip(boxes, scores, labels):
    if score >= confidence_threshold:
        detection_count += 1

        # Convert label to integer for dictionary lookup
        label_id = label.item()

        # Get class name using dictionary lookup (handles gaps correctly)
        class_name = COCO_INSTANCE_CATEGORY_NAMES.get(label_id, f'unknown_class_{label_id}')

        # Get object bounding box
        x1, y1, x2, y2 = map(int, box)

        # Print detection outputs with bounding box
        print(f'Detection {detection_count}: {class_name}'
              f'(ID: {label_id}) - Confidence: {score:.3f}')
        print(f'  Box: [{x1}, {y1}, {x2}, {y2}]')

        # Draw bounding box
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw label and score
        label_text = f'{class_name}: {score:.2f}'
        cv2.putText(image_bgr, label_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

print(f'Total detections above 0.5 confidence: {detection_count}')

# Show result using matplotlib
plt.figure(figsize=(10, 8))
plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('FCOS Object Detection')
plt.show()
