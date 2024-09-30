import random
import shutil
import sys
import os

from ultralytics import YOLO


def partition_validation(dataset_path):
    train_path = os.path.join(dataset_path, "train")
    valid_path = os.path.join(dataset_path, "valid")

    train_images_path = os.path.join(train_path, "images")
    train_labels_path = os.path.join(train_path, "labels")

    valid_images_path = os.path.join(valid_path, "images")
    valid_labels_path = os.path.join(valid_path, "labels")

    if os.path.exists(train_path) and os.path.exists(valid_path):
        print("Train and validation paths already exist. Stopping the process.")
        sys.exit()

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(train_images_path, exist_ok=True)
    os.makedirs(train_labels_path, exist_ok=True)
    os.makedirs(valid_path, exist_ok=True)
    os.makedirs(valid_images_path, exist_ok=True)
    os.makedirs(valid_labels_path, exist_ok=True)

    dataset_images = f"{dataset_path}\\export\\images"
    dataset_labels = f"{dataset_path}\\export\\labels"

    images = [f for f in os.listdir(dataset_images) if f.endswith(('.png', '.jpg'))]
    labels = [f.replace(".jpg", ".txt").replace(".png", ".txt") for f in images]

    data = list(zip(images, labels))

    random.shuffle(data)

    split_index = int(len(data) * 0.8)

    train_data = data[:split_index]
    valid_data = data[split_index:]

    for img, lbl in train_data:
        shutil.move(os.path.join(dataset_images, img), os.path.join(train_images_path, img))
        shutil.move(os.path.join(dataset_labels, lbl), os.path.join(train_labels_path, lbl))

    for img, lbl in valid_data:
        shutil.move(os.path.join(dataset_images, img), os.path.join(valid_images_path, img))
        shutil.move(os.path.join(dataset_labels, lbl), os.path.join(valid_labels_path, lbl))


if __name__ == "__main__":
    partition_validation("NO2-Dataset")

    model = YOLO('YOLOv8/yolov8s/yolov8s.pt')
    data_yaml_path = 'D:\\Fakultet\\VIProekt\\Proekt-VI\\NO2-Dataset\\data.yaml'

    model.train(
        data=data_yaml_path,
        epochs=12,
        imgsz=800,
        batch=4,
    )
