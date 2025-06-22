# People Detection & Tracking System 🕺👫

Программа для детекции и трекинга людей в видео с использованием YOLOv8. Создает видео с визуализацией детекций, уникальными ID и статистикой.

![Пример работы](demo.gif) <!-- Замените на реальный пример -->

## 🔍 Основные возможности

- 🎯 **Точная детекция** с YOLOv8 (поддержка всех версий моделей)
- 🆔 **Трекинг** с уникальными ID (алгоритм SORT)
- 📊 **Статистика** по количеству людей в кадре
- 🖼️ **Визуализация**:
  - Зеленые bounding boxes
  - Метки с ID и уверенностью
  - Счетчик людей в реальном времени
- ⚡ **Оптимизация** для CPU/GPU
- 📝 **Подробное логирование** процесса

## 🛠 Установка и настройка

### Требования
- Python 3.8+
- NVIDIA GPU (опционально, для CUDA ускорения)

### Установка
1. Клонируйте репозиторий:
```bash
git clone https://github.com/Apahc/People_video_detection.git
cd People_video_detection
```
2. Установите зависимости:

```bash
pip install -r requirements.txt
```
3. (Опционально) Для GPU ускорения:

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## 🚀 Использование
Базовый запуск
```bash
python people_detection_frame_by_frame.py --input crowd.mp4 --output output.mp4
```
Параметры запуска

Аргумент	Описание	По умолчанию

--input	Путь к входному видео	crowd.mp4

--output	Путь для выходного видео	output.mp4

--model	Модель YOLOv8 (n, s, m, l, x)	yolov8m.pt

--conf	Порог уверенности (0.0-1.0)	0.5

--iou	Порог IoU для NMS (0.0-1.0)	0.5

Примеры
```bash
# С пользовательскими параметрами
python people_detection_frame_by_frame.py --input video.mp4 --output result.mp4 --model yolov8l.pt --conf 0.6

# Только CPU
CUDA_VISIBLE_DEVICES="" python people_detection_frame_by_frame.py
```

## ⚙️ Технические детали

Производительность

## Аппаратура и Производительность

| Аппаратура | Разрешение | FPS       |
| :---------- | :--------- | :-------- |
| CPU (i7)    | 1280x720   | 4-8       |
| GPU (RTX 3060) | 1920x1080 | 25-35     |


## Модели YOLOv8 и Рекомендации

| Модель  | mAP@0.5 | Размер (MB) | Рекомендация       |
| :------- | :------ | :---------- | :------------------ |
| nano     | 0.58    | 6.2         | Для CPU             |
| small    | 0.62    | 21          | Баланс              |
| medium   | 0.65    | 49          | По умолчанию         |
| large    | 0.67    | 83          | Для точности         |
| xlarge   | 0.68    | 130         | Для плотных сцен     |

## 💡 Советы по улучшению
Для плотных толп:

```bash
python people_detection_frame_by_frame.py --model yolov8l.pt --conf 0.6 --iou 0.4
```
Для реального времени:

# Уменьшите разрешение
frame = cv2.resize(frame, (1280, 720))

# Экспорт в ONNX
model.export(format='onnx')
Дополнительный анализ:

# Генерация heatmap
heatmap = generate_heatmap(detections)
frame = overlay_heatmap(frame, heatmap)
