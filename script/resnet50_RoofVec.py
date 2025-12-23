import os
import json
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ------------- Dataset -------------------

class CitiesDataset(Dataset):
    def __init__(self, img_dir, annot_dir, file_list):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.images = file_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        annot_path = os.path.join(self.annot_dir, img_name.replace('.jpg', '.json'))

        img = Image.open(img_path).convert("RGB")
        img = F.to_tensor(img)

        with open(annot_path, 'r') as f:
            polygons = json.load(f)

        masks = []
        height, width = img.shape[1:]
        for poly in polygons:
            mask = np.zeros((height, width), dtype=np.uint8)
            poly_np = np.array(poly, np.int32)
            cv2.fillPoly(mask, [poly_np], 1)
            masks.append(mask)

        if len(masks) == 0:
            masks = np.zeros((0, height, width), dtype=np.uint8)
        else:
            masks = np.stack(masks, axis=0)

        masks = torch.as_tensor(masks, dtype=torch.uint8)

        num_objs = masks.shape[0]
        boxes = []
        for mask in masks:
            pos = torch.nonzero(mask, as_tuple=True)
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                xmin, ymin, xmax, ymax = 0, 0, 0, 0
            else:
                xmin = torch.min(pos[1]).item()
                xmax = torch.max(pos[1]).item()
                ymin = torch.min(pos[0]).item()
                ymax = torch.max(pos[0]).item()
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks
        }

        return img, target

# ------------- Model -------------------

def get_model(num_classes):
    weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT  
    model = maskrcnn_resnet50_fpn(weights=weights)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model

# ------------- Training Loop -------------------

def train_model(model, train_loader, val_loader, device, num_epochs=50):
    optimizer = optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    best_mean_iou = 0
    patience = 15
    counter = 0

    lr_history = []
    train_loss_history = []
    val_iou_history = []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # -------- Training --------
        model.train()
        total_loss = 0
        for imgs, targets in tqdm(train_loader, desc="Training"):
            imgs = [img.to(device) for img in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(imgs, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

        lr_scheduler.step()
        avg_loss = total_loss / len(train_loader)  

        # -------- Validation --------
        model.eval()
        tp, fp, total_gt = 0, 0, 0
        total_instance_ious = []

        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc="Validation"):
                imgs = [img.to(device) for img in imgs]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                outputs = model(imgs)

                for output, target in zip(outputs, targets):
                    pred_masks = (output['masks'].squeeze(1) > 0.5)
                    pred_scores = output['scores']
                    true_masks = target['masks'].bool()

                    num_gt = true_masks.shape[0]
                    total_gt += num_gt

                    if pred_masks.shape[0] == 0:
                        continue

                    matched_gt = set()
                    sorted_indices = torch.argsort(pred_scores, descending=True)
                    pred_masks = pred_masks[sorted_indices]

                    for pred_mask in pred_masks:
                        if true_masks.shape[0] == 0:
                            fp += 1
                            continue

                        inter = (pred_mask & true_masks).float().sum(dim=(1, 2))
                        union = (pred_mask | true_masks).float().sum(dim=(1, 2))
                        iou_per_gt = inter / (union + 1e-6)

                        best_iou_value, best_gt_idx = torch.max(iou_per_gt, dim=0)

                        if best_iou_value.item() >= 0.5 and best_gt_idx.item() not in matched_gt:
                            tp += 1
                            matched_gt.add(best_gt_idx.item())
                            total_instance_ious.append(best_iou_value.item())
                        else:
                            fp += 1

        mean_iou = np.mean(total_instance_ious) if total_instance_ious else 0.0
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / total_gt if total_gt > 0 else 0.0

        print(f"Epoch {epoch+1}:")
        print(f"Training Loss = {avg_loss:.4f}")  
        print(f"Validation ➔ Mean IoU = {mean_iou:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}")
        print("-" * 50)

        train_loss_history.append(avg_loss)
        val_iou_history.append(mean_iou)
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        if mean_iou > best_mean_iou:
            best_mean_iou = mean_iou
            counter = 0
            filename = f"best_model_epoch{epoch+1}_iou{mean_iou:.4f}_RoofVec.pth"  
            torch.save(model.state_dict(), filename)
            print(f"Saved best model ➔ {filename}")
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered!")
                break

    # -------- After Training: Plot Graphs and Save --------
    epochs = range(1, len(train_loss_history) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr_history, marker='o')
    plt.title("Learning Rate Schedule")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("learning_rate_schedule_RoofVec.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss_history, color="tab:red", marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss_curve_RoofVec.png")
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_iou_history, color="tab:blue", marker='x')
    plt.title("Validation Mean IoU")
    plt.xlabel("Epoch")
    plt.ylabel("Mean IoU")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_mean_iou_curve_RoofVec.png")
    plt.show()

# ------------- Main -------------------

if __name__ == "__main__":
    # Please update the path below to match your local file system
    img_dir = r"/scratch/hycheng/Thesis/new_Maskrcnn_train/data/RoofVec/train/rgb" 
    annot_dir = r"/scratch/hycheng/Thesis/new_Maskrcnn_train/data/RoofVec/train/RoofVec_JSON"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_images = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])

    subset_ratio = 1
    subset_size = max(1, int(len(all_images) * subset_ratio))
    subset_images = all_images[:subset_size]

    train_imgs, val_imgs = train_test_split(subset_images, test_size=0.25, random_state=42)

    train_dataset = CitiesDataset(img_dir, annot_dir, train_imgs)
    val_dataset = CitiesDataset(img_dir, annot_dir, val_imgs)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(num_classes=2)
    model.to(device)

    train_model(model, train_loader, val_loader, device, num_epochs=100)
