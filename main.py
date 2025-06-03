from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection import retinanet_resnet50_fpn
from PIL import Image, ImageDraw, ImageFont
from torchvision.utils import draw_bounding_boxes
import os
from torchvision.models import resnet50
import json
import urllib.request
import time


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

def download_imagenet_labels():
    # 1. Define the raw URL in the keras-vis repository
    url = (
        "https://raw.githubusercontent.com/raghakot/keras-vis/"
        "master/resources/imagenet_class_index.json"
    )

    # 2. Download and parse the JSON
    with urllib.request.urlopen(url) as response:
        data = response.read().decode("utf-8")
        idx_to_label = json.loads(data)

    # 3. Convert to a simple list: idx 0 → label, idx 1 → label, …, idx 999 → label
    imagenet_labels = [idx_to_label[str(i)][1] for i in range(1000)]

    print("Label 0:", imagenet_labels[0])  # e.g., "tench"
    return imagenet_labels


class Detection:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device ", self.device, " is available")

        # model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.model = retinanet_resnet50_fpn(pretrained=True).to(self.device)
        self.model.eval()

        # warmup
        warmup_image = torch.ones([3, 256, 256]).type(torch.float32).to(self.device)
        with torch.no_grad():
            _ = self.model([warmup_image])

        self.imagenet_labels = download_imagenet_labels()

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")

        # Resize if too big (optional)
        max_size = 1024
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h))

        image_tensor = transforms.ToTensor()(image).to(self.device)
        return image_tensor

    def detect_borders(self, image_tensor):
        start = time.perf_counter()
        with torch.no_grad():
            outputs = self.model([image_tensor])
        end = time.perf_counter()
        elapsed = end - start  # seconds per frame
        fps = 1.0 / elapsed
        print(f"Inner detection performance: {elapsed:.3f} s → {fps:.1f} FPS")

        # outputs is a list (length=1) of dicts
        detections = outputs[0]
        boxes = detections["boxes"]        # shape: [N, 4]
        labels = detections["labels"]      # shape: [N]
        scores = detections["scores"]      # shape: [N]

        # 5. Filter by confidence threshold
        threshold = 0.5
        kept = scores >= threshold
        filtered_boxes = boxes[kept].cpu().tolist()
        filtered_labels = labels[kept].cpu().tolist()
        filtered_scores = scores[kept].cpu().tolist()

        return filtered_boxes, filtered_labels, filtered_scores

    def build_naive_summary(self, boxes, labels, scores):
        vision_summary = []
        for box, lbl, sc in zip(boxes, labels, scores):
            class_name = COCO_CLASSES[lbl]
            print(class_name)
            xmin, ymin, xmax, ymax = [int(b) for b in box]
            vision_summary.append(
                f"{class_name} at ({xmin},{ymin})-({xmax},{ymax}), score {sc:.2f}"
            )
        # vision_summary is now a list of strings like "person at (50,320)-(120,700), score 0.87"
        return vision_summary
    
    def save_image_with_boxes(self, image, boxes, labels, scores):
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


def main():
    detector = Detection()

    names = ["images/cats_house.jpg", "images/city.jpg", "images/summer.jpg"]

    images = []
    for name in names:
        image = detector.load_image("images/cats_house.jpg")
        images.append(image)

    for image in images:
        boxes, labels, scores = detector.detect_borders(image)

    print(detector.build_naive_summary(boxes, labels, scores))


main()


'''
# 7. Try to obtain better labels
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225]),
])


imagenet_labels = download_imagenet_labels()

cropped_classifier = resnet50(pretrained=True).eval().to(device)

# ─── D. Crop + classify each box ───────────────────────────────────────────────
for i, box in enumerate(filtered_boxes):
    xmin, ymin, xmax, ymax = [int(b) for b in box]
    crop_images = image.crop((xmin, ymin, xmax, ymax))

    input_tensor = preprocess(crop_images).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cropped_classifier(input_tensor)
        probs  = torch.softmax(logits[0], dim=0)

    top1_idx = torch.argmax(probs).item()
    top1_conf = probs[top1_idx].item()
    top1_label = imagenet_labels[top1_idx]

    print(f"Box {i}: ({xmin},{ymin},{xmax},{ymax}) → '{top1_label}' ({top1_conf:.2f})")
'''
