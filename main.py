import cv2
import numpy as np
import torch
from lane_detection_model.lane_detector import ENet
from ultralytics import YOLO


def load_lane_model(model_path):
    enet_model = ENet(2, 4)
    enet_model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
    enet_model.eval()
    return enet_model


def load_yolo_model(model_path):
    return YOLO(model_path)


def preprocess_frame_for_lane_detection(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_image = cv2.resize(gray_frame, (512, 256))
    input_image = input_image[..., None]
    input_tensor = torch.from_numpy(input_image).float().permute(2, 0, 1)
    return input_tensor, input_image


def detect_lanes(enet_model, input_tensor):
    with torch.no_grad():
        binary_logits, instance_logits = enet_model(input_tensor.unsqueeze(0))
    binary_seg = torch.argmax(binary_logits, dim=1).squeeze().numpy()
    return binary_seg


def postprocess_lane_detection(binary_seg, input_image, frame_size):
    binary_seg_grayscale = np.zeros_like(input_image[:, :, 0])
    binary_seg_grayscale[binary_seg == 1] = 255
    binary_seg_grayscale = cv2.resize(binary_seg_grayscale, (frame_size[0], frame_size[1]))
    return binary_seg_grayscale


def overlay_lane_mask(frame, binary_seg_grayscale):
    output_image = frame.copy()
    output_image[:, :, 0] = cv2.addWeighted(output_image[:, :, 0], 0.7, binary_seg_grayscale, 0.3, gamma=0)
    return output_image


def detect_objects(yolo_model, frame):
    results = yolo_model(frame, conf=0.5)
    annotated_frame = results[0].plot(line_width=2)
    return annotated_frame


def process_video_frame(frame, enet_model, yolo_model, frame_size):
    input_tensor, input_image = preprocess_frame_for_lane_detection(frame)
    binary_seg = detect_lanes(enet_model, input_tensor)
    binary_seg_grayscale = postprocess_lane_detection(binary_seg, input_image, frame_size)
    lane_overlay_frame = overlay_lane_mask(frame, binary_seg_grayscale)
    annotated_frame = detect_objects(yolo_model, frame)
    combined_output = cv2.addWeighted(lane_overlay_frame, 0.5, annotated_frame, 0.7, gamma=0)
    return combined_output


def process_and_save_video(input_video_path, output_video_path, enet_model, yolo_model):
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), 10, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        combined_output = process_video_frame(frame, enet_model, yolo_model, (frame_width, frame_height))

        out.write(combined_output)

        cv2.imshow('Lane and Object Detection', combined_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    input_video_path = "test_videos/example_video10.mp4"
    output_video_path = 'result_videos/final_result_video.avi'
    lane_model_path = 'lane_detection_model/ENET.pth'
    yolo_model_path = "runs/detect/train2/weights/best.pt"

    enet_model = load_lane_model(lane_model_path)
    yolo_model = load_yolo_model(yolo_model_path)

    process_and_save_video(input_video_path, output_video_path, enet_model, yolo_model)
