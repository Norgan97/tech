import cv2
import torch


# Загрузка модели YOLOv5
model_path = 'best.pt'
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path)

# Чтение видео
input_video_path = 'crowd.mp4'
cap = cv2.VideoCapture(input_video_path)

# Получение свойств видео
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Настройка выходного видео
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output_video.mp4', fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Преобразование кадра для модели
    results = model(frame)
    
    # Отрисовка результатов на кадре
    annotated_frame = results.render()[0]
    
    # Сохранение обработанного кадра
    out.write(annotated_frame)

# Освобождение ресурсов
cap.release()
out.release()
cv2.destroyAllWindows()