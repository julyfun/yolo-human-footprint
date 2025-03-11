import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


class PersonTracker:
    def __init__(self, yolo_model="yolov8n.pt", video_path=None):
        """初始化跟踪器"""
        # 加载YOLO模型
        self.model = YOLO(yolo_model)
        self.video_path = video_path
        # 用于存储每个ID的轨迹
        self.tracks = defaultdict(list)
        # 轨迹颜色映射
        self.color_map = {}

    def generate_color(self, idx):
        """为每个跟踪ID生成唯一的颜色"""
        if idx not in self.color_map:
            self.color_map[idx] = tuple(map(int, np.random.randint(0, 255, size=3)))
        return self.color_map[idx]

    def process_video(self, output_path="output.mp4"):
        """处理视频并绘制轨迹"""
        if not self.video_path:
            raise ValueError("请提供视频路径")

        cap = cv2.VideoCapture(self.video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 设置输出视频
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        frame_count = 0

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            print(f"处理第 {frame_count} 帧")

            # 使用YOLO进行检测和跟踪 (conf=0.3 为置信度阈值)
            results = self.model.track(
                frame, persist=True, classes=0, conf=0.3
            )  # 只跟踪人 (class 0)

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # 更新轨迹并绘制边界框
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    # 使用底部中心点作为轨迹点
                    center_x = (x1 + x2) // 2
                    bottom_y = y2
                    self.tracks[track_id].append((center_x, bottom_y))

                    # 绘制边界框
                    color = self.generate_color(track_id)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    # 在边界框上方显示ID
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            # 在当前帧上绘制所有轨迹
            for track_id, points in self.tracks.items():
                if len(points) > 1:  # 至少需要两个点才能绘制线条
                    color = self.generate_color(track_id)
                    # 绘制轨迹线
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i - 1], points[i], color, 2)
                    # 在最新位置显示ID
                    cv2.putText(
                        frame,
                        f"ID: {track_id}",
                        (points[-1][0], points[-1][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

            # 保存到输出视频
            out.write(frame)

            # 显示处理后的帧 (可选，处理大视频时可能会减慢速度)
            cv2.imshow("Tracking 2", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        print(f"视频已处理完成，保存为 {output_path}")


# 使用示例
if __name__ == "__main__":
    tracker = PersonTracker(
        yolo_model="yolo12x.pt",  # 使用小型模型，您可以根据需要替换为其他版本
        video_path="003.mp4",  # 替换为您的视频路径
    )
    tracker.process_video("output-003.mp4")
