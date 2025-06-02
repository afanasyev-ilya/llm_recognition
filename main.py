from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import draw_bounding_boxes

COCO_CLASSES = [
    "__background__",
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic_light", "fire_hydrant", "stop_sign", "parking_meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports_ball", "kite", "baseball_bat", "baseball_glove",
    "skateboard", "surfboard", "tennis_racket", "bottle", "wine_glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot_dog", "pizza", "donut", "cake", "chair", "couch",
    "potted_plant", "bed", "dining_table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell_phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy_bear",
    "hair_drier", "toothbrush"
]

def save_image_with_boxes(image, boxes, labels, scores):
    # 7. Save recognition results to file
    original_image = image
    draw_img = original_image.copy()
    draw = ImageDraw.Draw(draw_img)

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except IOError:
        font = ImageFont.load_default()

    for box, lbl, sc in zip(boxes, labels, scores):
        class_name = COCO_CLASSES[lbl]
        xmin, ymin, xmax, ymax = [box[0], box[1], box[2], box[3]]

        # Build label text
        label_text = f"{class_name} {sc:.2f}"

        # Draw a filled red rectangle behind the text
        margin = 2
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

        # Draw text in white on top of that background
        text_position = (xmin, ymin)
        draw.text(text_position, label_text, fill="white", font=font)

    # 4. Save the result
    output_path = "images/annotated_image.jpg"
    draw_img.save(output_path)
    print(f"Saved annotated image to {output_path!r}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device ", device, " is available")

# 1. Load image
img_path = "images/cats_house.jpg"
image = Image.open(img_path).convert("RGB")

# 2. Resize if too big (optional)
max_size = 1024
w, h = image.size
if max(w, h) > max_size:
    scale = max_size / float(max(w, h))
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h))

# 3. Transform to tensor
image_tensor = transforms.ToTensor()(image).to(device)

# 4. Load pretrained detector
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval()

with torch.no_grad():
    outputs = model([image_tensor])
# outputs is a list (length=1) of dicts
detections = outputs[0]
boxes = detections["boxes"]        # shape: [N, 4]
labels = detections["labels"]      # shape: [N]
scores = detections["scores"]      # shape: [N]

print(len(COCO_CLASSES))

# 5. Filter by confidence threshold
threshold = 0.5
kept = scores >= threshold
filtered_boxes = boxes[kept].cpu().tolist()
filtered_labels = labels[kept].cpu().tolist()
filtered_scores = scores[kept].cpu().tolist()

# 6. Build structured summary
vision_summary = []
for box, lbl, sc in zip(filtered_boxes, filtered_labels, filtered_scores):
    class_name = COCO_CLASSES[lbl]
    print(class_name)
    xmin, ymin, xmax, ymax = [int(b) for b in box]
    vision_summary.append(
        f"{class_name} at ({xmin},{ymin})-({xmax},{ymax}), score {sc:.2f}"
    )
print(vision_summary)
# vision_summary is now a list of strings like "person at (50,320)-(120,700), score 0.87"

# visualize
save_image_with_boxes(image, filtered_boxes, filtered_labels, filtered_scores)


