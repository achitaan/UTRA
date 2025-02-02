from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('hand_wash_stage_and_correctness_model.h5')

# Load and preprocess an image
img_path = 'path_to_test_image.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions
predictions = model.predict(img_array)
stage_prediction = np.argmax(predictions[0])  # Predicted stage
correctness_prediction = predictions[1][0]   # Correctness probability

print(f'Predicted Stage: {stage_prediction}')
print(f'Correctness Probability: {correctness_prediction}')