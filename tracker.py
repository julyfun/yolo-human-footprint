import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict


class PersonTracker:
    def __init__(
        self, yolo_model="yolov8n.pt", video_path=None, max_trajectory_length=50
    ):
        """初始化跟踪器"""
        # 加载YOLO模型
        self.model = YOLO(yolo_model)
        self.video_path = video_path
        # 用于存储每个ID的轨迹
        self.tracks = defaultdict(list)
        # 轨迹颜色映射
        self.color_map = {}
        self.min_hits = 3  # 最小检测次数
        self.max_age = 10  # 最大丢失帧数
        self.track_history = defaultdict(dict)  # 跟踪历史
        self.features = {}  # 存储每个ID的特征向量
        self.feature_memory = 50  # 特征记忆帧数
        self.max_trajectory_length = max_trajectory_length  # 轨迹最大长度

    def generate_color(self, idx):
        """为每个跟踪ID生成唯一的颜色"""
        if idx not in self.color_map:
            self.color_map[idx] = tuple(map(int, np.random.randint(0, 255, size=3)))
        return self.color_map[idx]

    def update_tracks(self, track_id, box):
        """更新轨迹，并限制长度"""
        if track_id not in self.track_history:
            self.track_history[track_id] = {"hits": 1, "age": 0, "last_box": box}
        else:
            self.track_history[track_id]["hits"] += 1
            self.track_history[track_id]["age"] = 0
            self.track_history[track_id]["last_box"] = box

    def predict_next_position(self, track_points, num_points=5):
        """简单的线性预测"""
        if len(track_points) < num_points:
            return None
        recent_points = track_points[-num_points:]
        dx = (recent_points[-1][0] - recent_points[0][0]) / (num_points - 1)
        dy = (recent_points[-1][1] - recent_points[0][1]) / (num_points - 1)
        next_x = int(recent_points[-1][0] + dx)
        next_y = int(recent_points[-1][1] + dy)
        return (next_x, next_y)

    def extract_features(self, frame, box):
        """提取人物特征"""
        x1, y1, x2, y2 = box
        person_img = frame[y1:y2, x1:x2]
        # 调整大小为统一尺寸
        person_img = cv2.resize(person_img, (64, 128))
        # 计算简单的颜色直方图作为特征
        features = cv2.calcHist(
            [person_img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]
        )
        features = cv2.normalize(features, features).flatten()
        return features

    def match_features(self, features):
        """匹配特征找到最相似的ID"""
        best_match = None
        min_dist = float("inf")
        for track_id, stored_features in self.features.items():
            dist = np.linalg.norm(features - stored_features)
            if dist < min_dist and dist < 0.3:  # 阈值可调
                min_dist = dist
                best_match = track_id
        return best_match

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

            # 修改跟踪参数，只使用有效的YOLO参数
            results = self.model.track(
                frame,
                persist=True,  # 启用跨帧跟踪
                classes=0,  # 只跟踪人类
                conf=0.5,  # 置信度阈值
                iou=0.5,  # IOU阈值
            )

            if results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)

                # 更新所有轨迹的年龄
                for tid in list(self.track_history.keys()):
                    self.track_history[tid]["age"] += 1
                    # 删除过旧的轨迹
                    if self.track_history[tid]["age"] > self.max_age:
                        del self.track_history[tid]
                        if tid in self.tracks:
                            del self.tracks[tid]

                # 更新轨迹
                for box, track_id in zip(boxes, track_ids):
                    self.update_tracks(track_id, box)
                    # 只有当跟踪稳定时才更新轨迹
                    if self.track_history[track_id]["hits"] >= self.min_hits:
                        x1, y1, x2, y2 = box
                        # 使用底部中心点作为轨迹点
                        center_x = (x1 + x2) // 2
                        bottom_y = y2
                        self.tracks[track_id].append((center_x, bottom_y))

                        # 限制轨迹长度
                        if len(self.tracks[track_id]) > self.max_trajectory_length:
                            self.tracks[track_id] = self.tracks[track_id][
                                -self.max_trajectory_length :
                            ]

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
                    # 绘制历史轨迹
                    for i in range(1, len(points)):
                        cv2.line(frame, points[i - 1], points[i], color, 2)

                    # 预测并绘制未来轨迹
                    future_points = self.predict_next_positions(points)
                    if future_points:
                        # 绘制预测轨迹（虚线）
                        last_point = points[-1]
                        for future_point in future_points:
                            self.draw_dashed_line(
                                frame, last_point, future_point, color, 2
                            )
                            last_point = future_point
                            # 在预测点画小圆点
                            cv2.circle(frame, future_point, 3, color, -1)

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

    def draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_length=30):
        """绘制虚线"""
        dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
        pts = []
        for i in np.arange(0, dist, dash_length):
            r = i / dist
            x = int((pt1[0] * (1 - r) + pt2[0] * r))
            y = int((pt1[1] * (1 - r) + pt2[1] * r))
            pts.append((x, y))

        # 分段绘制实线，形成虚线效果
        for i in range(len(pts) - 1):
            if i % 2 == 0:
                cv2.line(img, pts[i], pts[i + 1], color, thickness)

    def predict_next_positions(self, track_points, num_points=10, future_steps=15):
        """预测多个未来位置点，使用平滑处理"""
        if len(track_points) < num_points:
            return None

        # 使用更多的历史点来计算平均速度
        recent_points = track_points[-num_points:]

        # 计算多段速度并平均
        velocities_x = []
        velocities_y = []
        for i in range(1, len(recent_points)):
            dx = recent_points[i][0] - recent_points[i - 1][0]
            dy = recent_points[i][1] - recent_points[i - 1][1]
            velocities_x.append(dx)
            velocities_y.append(dy)

        # 使用移动平均计算稳定的速度
        if len(velocities_x) >= 3:
            dx = sum(velocities_x[-3:]) / 3  # 使用最近3个速度的平均值
            dy = sum(velocities_y[-3:]) / 3
        else:
            dx = sum(velocities_x) / len(velocities_x)
            dy = sum(velocities_y) / len(velocities_y)

        # 速度限制，防止预测过大的变化
        max_speed = 10  # 最大速度限制
        dx = max(min(dx, max_speed), -max_speed)
        dy = max(min(dy, max_speed), -max_speed)

        # 生成预测点
        future_points = []
        last_x, last_y = recent_points[-1]

        for i in range(future_steps):
            # 使用衰减因子，让远期预测更保守
            decay = 0.95**i  # 速度衰减因子
            next_x = int(last_x + dx * decay)
            next_y = int(last_y + dy * decay)
            future_points.append((next_x, next_y))
            last_x, last_y = next_x, next_y

        return future_points


# 使用示例
if __name__ == "__main__":
    tracker = PersonTracker(
        yolo_model="yolo12x.pt",
        video_path="003.mp4",
        max_trajectory_length=50,  # 只保留最近40帧的轨迹
    )
    tracker.process_video("output-003.mp4")
