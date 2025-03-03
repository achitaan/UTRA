import os
import cv2
import json
import numpy as np
import pandas as pd
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Define paths
videos_dir = r"C:\Programming\UTRA\Train-Hand-Model\dataset\Videos"
annotations_dir = r"C:\Programming\UTRA\Train-Hand-Model\dataset\Annotations"

# Step 2: Load and preprocess data
def load_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def load_annotations(annotation_path):
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    return annotations

def preprocess_frames(frames, target_size=(224, 224)):
    resized_frames = [cv2.resize(frame, target_size) for frame in frames]
    normalized_frames = np.array(resized_frames, dtype=np.float32) / 255.0
    return normalized_frames

video_files = [f for f in os.listdir(videos_dir) if f.endswith('.mp4')]
all_frames = []
all_labels = []

print(f"asjdaklsd {len(video_files)}")

for video_file in video_files:
    video_path = os.path.join(videos_dir, video_file)
    frames = load_video_frames(video_path)

    annotation_file = video_file.replace('.mp4', '.json')
    annotation_path = None
    for root, dirs, files in os.walk(annotations_dir):
        if annotation_file in files:
            annotation_path = os.path.join(root, annotation_file)
            break

    if annotation_path is None:
        print(f"Skipping {video_file}: Annotation file not found.")
        continue

    annotations = load_annotations(annotation_path)

    frame_labels = [label['code'] for label in annotations['labels']]

    if len(frames) != len(frame_labels):
        print(f"Skipping {video_file}: Frame count mismatch. Frames: {len(frames)}, Labels: {len(frame_labels)}")
        continue

    # Preprocess frames
    normalized_frames = preprocess_frames(frames)

    all_frames.extend(normalized_frames)
    all_labels.extend(frame_labels)

all_frames = np.array(all_frames, dtype=np.float32)
all_labels = np.array(all_labels, dtype=np.int32)


print(f"Total frames: {len(all_frames)}")
print(f"Total labels: {len(all_labels)}")

if len(all_frames) == 0 or len(all_labels) == 0:
    raise ValueError("No data found. Check the dataset paths and file formats.")
else:
    X_train, X_val, y_train, y_val = train_test_split(all_frames, all_labels, test_size=0.2, random_state=42)

base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(8, activation='softmax')(x)  # 8 classes for movement codes (0–7)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss}')
print(f'Validation Accuracy: {accuracy}')

model.save('hand_wash_movement_model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(10)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
