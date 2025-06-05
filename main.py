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
    def __init__(self, use_half = True):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device ", self.device, " is available")

        self.use_half = use_half
        if self.device.type != 'cuda':
            self.use_half = False

        # model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device)
        self.borders_detector = retinanet_resnet50_fpn(pretrained=True).to(self.device)
        self.borders_detector.eval()
        if self.use_half:
            self.borders_detector = self.borders_detector.half()

        # warmup
        model_dtype = next(self.borders_detector.parameters()).dtype
        warmup_image = torch.ones([3, 256, 256]).type(model_dtype).to(self.device)
        with torch.no_grad():
            _ = self.borders_detector([warmup_image])

        self.imagenet_labels = download_imagenet_labels()

        self.cropped_classifier = resnet50(pretrained=True).eval().to(self.device)
        if self.use_half:
            self.cropped_classifier = self.borders_detector.half()

    def load_image(self, img_path):
        image = Image.open(img_path).convert("RGB")

        # Resize if too big (optional)
        max_size = 1024
        w, h = image.size
        if max(w, h) > max_size:
            scale = max_size / float(max(w, h))
            new_w, new_h = int(w * scale), int(h * scale)
            image = image.resize((new_w, new_h))
    
        return image

    def img2tensor(self, image):
        model_dtype = next(self.borders_detector.parameters()).dtype
        tensor = transforms.ToTensor()(image).to(self.device).to(dtype=model_dtype)
        return tensor

    def detect_borders(self, image):
        image_tensor = self.img2tensor(image)

        with torch.no_grad():
            outputs = self.borders_detector([image_tensor])

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

    def classify_cropped_parts(self, image, boxes):
        # 7. Try to obtain better labels
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std =[0.229, 0.224, 0.225]),
        ])

        labels = []
        conf_levels = []
        for i, box in enumerate(boxes):
            xmin, ymin, xmax, ymax = [int(b) for b in box]
            crop_images = image.crop((xmin, ymin, xmax, ymax))

            input_tensor = preprocess(crop_images).unsqueeze(0).to(self.device)
            
            model_dtype = next(self.cropped_classifier.parameters()).dtype
            input_tensor = input_tensor.to(dtype=model_dtype)
            with torch.no_grad():
                logits = self.cropped_classifier(input_tensor)
                probs  = torch.softmax(logits[0], dim=0)

            top1_idx = torch.argmax(probs).item()
            top1_conf = probs[top1_idx].item()
            top1_label = self.imagenet_labels[top1_idx]
            labels.append(top1_label)
            conf_levels.append(top1_conf)
        return labels, conf_levels

    def build_summary(self, boxes, labels, scores):
        vision_summary = []
        #for box, lbl, sc in zip(boxes, labels, scores):
        #    xmin, ymin, xmax, ymax = [int(b) for b in box]
        #    vision_summary.append(
        #        f"{lbl} at ({xmin},{ymin})-({xmax},{ymax}), score {sc:.2f}"
        #    )
        i = 0
        summary = ""
        for box, lbl, sc in zip(boxes, labels, scores):
            summary += f"Box {i}: ({box[0]:.2f},{box[1]:.2f},{box[2]:.2f},{box[3]:.2f}) → '{lbl}' ({sc:.2f})\n"
            i += 1
        
        return summary
    
    def save_image_with_boxes(self, image, boxes, labels, scores, out_path):
        # 7. Save recognition results to file
        original_image = image
        draw_img = original_image.copy()
        draw = ImageDraw.Draw(draw_img)

        try:
            font = ImageFont.truetype("DejaVuSans.ttf", size=16)
        except IOError:
            font = ImageFont.load_default()

        for box, lbl, sc in zip(boxes, labels, scores):
            xmin, ymin, xmax, ymax = [box[0], box[1], box[2], box[3]]

            # Build label text
            label_text = f"{lbl} {sc:.2f}"

            # Draw a filled red rectangle behind the text
            margin = 2
            draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)

            # Draw text in white on top of that background
            text_position = (xmin, ymin)
            draw.text(text_position, label_text, fill="white", font=font)

        # 4. Save the result
        draw_img.save(out_path)



def main():
    detector = Detection(use_half = True)

    names = ["cats_house.jpg", "city.jpg", "summer.jpg"]
    input_names = [("images/" + name) for name in names]
    output_names = [("artifacts/" + name) for name in names]

    for in_path, out_path in zip(input_names, output_names):
        image = detector.load_image(in_path)

        start = time.perf_counter()
        boxes, _, _ = detector.detect_borders(image)
        box_labels, box_conf_levels = detector.classify_cropped_parts(image, boxes)
        end = time.perf_counter()
        elapsed = end - start  # seconds per frame
        fps = 1.0 / elapsed
        print(f"Full detection performance: {elapsed:.3f} s → {fps:.1f} FPS")

        summary = detector.build_summary(boxes, box_labels, box_conf_levels)
        #print(summary)

        detector.save_image_with_boxes(image, boxes, box_labels, box_conf_levels, out_path)


main()

