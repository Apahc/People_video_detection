import cv2
import os
import torch
import numpy as np
from ultralytics import YOLO
import argparse
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('detector.log'),
        logging.StreamHandler()
    ]
)

class PeopleDetector:
    """Класс для детекции и трекинга людей в видео с использованием YOLOv8."""

    def __init__(self, model_name='yolov8m.pt', conf=0.5, iou=0.5):
        """
        Инициализация детектора.

        Args:
            model_name (str): Название модели YOLOv8 (по умолчанию 'yolov8m.pt').
            conf (float): Порог уверенности для детекции (0.0–1.0).
            iou (float): Порог IoU для Non-Maximum Suppression (0.0–1.0).
        """
        self.class_name = 'person'
        self.class_id = 0  # ID класса 'person' в COCO
        self.conf = conf
        self.iou = iou
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Инициализация детектора: устройство={self.device}, модель={model_name}, conf={conf}, iou={iou}")
        self.model = self._load_model(model_name)

    def _load_model(self, model_name):
        """
        Загрузка модели YOLOv8.

        Args:
            model_name (str): Название файла модели.

        Returns:
            YOLO: Загруженная модель.

        Raises:
            RuntimeError: Если модель не удалось загрузить.
        """
        try:
            model = YOLO(model_name).to(self.device)
            return model
        except Exception as e:
            logging.error(f"Ошибка загрузки модели: {e}")
            raise RuntimeError(f"Ошибка загрузки модели: {e}")

    def process_video(self, input_path, output_path):
        """
        Обработка видеофайла с детекцией и трекингом людей.

        Args:
            input_path (str): Путь к входному видеофайлу.
            output_path (str): Путь для сохранения результата.

        Raises:
            FileNotFoundError: Если входной файл не существует.
            IOError: Если видео не удалось открыть или создать выходное видео.
        """
        if not os.path.exists(input_path):
            logging.error(f"Входной файл не найден: {input_path}")
            raise FileNotFoundError(f"Входной файл не найден: {input_path}")

        # Открытие видео
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            logging.error(f"Не удалось открыть видео: {input_path}")
            raise IOError(f"Не удалось открыть видео: {input_path}")

        # Параметры видео
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Видео: {input_path}, размер={width}x{height}, FPS={fps}, кадров={total_frames}")

        # Инициализация VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'H264') if os.name != 'nt' else cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            logging.error(f"Не удалось создать выходное видео: {output_path}")
            raise IOError(f"Не удалось создать выходное видео: {output_path}")

        # Обработка кадров
        frame_count = 0
        people_counts = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра
            frame, people_in_frame = self._process_frame(frame)
            people_counts.append(people_in_frame)

            # Сохранение кадра
            out.write(frame)
            frame_count += 1
            logging.info(f"Обработано: {frame_count}/{total_frames} ({frame_count / total_frames:.1%})")

        # Статистика
        if people_counts:
            avg_people = np.mean(people_counts)
            max_people = np.max(people_counts)
            logging.info(f"Статистика: среднее число людей={avg_people:.1f}, максимум={max_people}")

        # Освобождение ресурсов
        cap.release()
        out.release()
        logging.info(f"Обработка завершена. Результат сохранён в: {output_path}")

    def _process_frame(self, frame):
        """
        Обработка одного кадра: детекция, трекинг и отрисовка людей.

        Args:
            frame (numpy.ndarray): Входной кадр.

        Returns:
            tuple: (Кадр с отрисовкой, количество людей в кадре).
        """
        # Детекция и трекинг
        results = self.model.track(
            frame,
            classes=[self.class_id],
            conf=self.conf,
            iou=self.iou,
            persist=True,
            verbose=False
        )

        people_in_frame = 0
        for result in results:
            people_in_frame = len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                id = int(box.id) if box.id is not None else 0

                # Зелёный цвет для рамок и фона метки
                color = (0, 255, 0)

                # Отрисовка прямоугольника
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Метка с ID и уверенностью
                label = f"{self.class_name} {id}: {conf:.2f}"
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

                # Полупрозрачный фон для текста
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1 - 20), (x1 + w, y1), color, -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )

        # Подсчёт людей в кадре
        cv2.putText(
            frame,
            f"People: {people_in_frame}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        return frame, people_in_frame

def main():
    """Точка входа в программу."""
    parser = argparse.ArgumentParser(description='Детекция и трекинг людей в видео с использованием YOLOv8.')
    parser.add_argument('--input', default='crowd.mp4', help='Путь к входному видеофайлу (по умолчанию: crowd.mp4).')
    parser.add_argument('--output', default='output.mp4', help='Путь для выходного видео (по умолчанию: output.mp4).')
    parser.add_argument('--model', default='yolov8m.pt', help='Модель YOLOv8 (по умолчанию: yolov8m.pt).')
    parser.add_argument('--conf', type=float, default=0.5, help='Порог уверенности (по умолчанию: 0.5).')
    parser.add_argument('--iou', type=float, default=0.5, help='Порог IoU для NMS (по умолчанию: 0.5).')
    args = parser.parse_args()

    logging.info(
        f"Запуск программы: модель={args.model}, вход={args.input}, "
        f"выход={args.output}, conf={args.conf}, iou={args.iou}"
    )

    try:
        detector = PeopleDetector(args.model, args.conf, args.iou)
        detector.process_video(args.input, args.output)

        # Анализ результатов
        logging.info("\nАнализ результатов:")
        logging.info("1. Для повышения точности используйте yolov8l.pt или обучите модель на кастомном датасете.")
        logging.info("2. Настройте порог уверенности (--conf) для фильтрации ложных детекций.")
        logging.info("3. Для плотных толп добавьте heatmap-анализ зон скопления.")
        logging.info("4. Для реального времени используйте batch processing и оптимизацию модели (ONNX/TensorRT).")

    except Exception as e:
        logging.error(f"Ошибка выполнения: {e}")
        print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()