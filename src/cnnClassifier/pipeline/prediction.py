import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        try:
            # Load model
            model = load_model(os.path.join("model", "model.h5"))

            # Load and preprocess image
            img_path = self.filename
            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = img_array.astype('float32') / 255  # Scale pixel values to [0, 1]

            # Predict
            predicted_class_index = np.argmax(model.predict(img_array), axis=1)[0]

            # Interpret prediction
            if predicted_class_index == 1:
                prediction = 'Tumor'
            else:
                prediction = 'Normal'

            return [{"image": prediction}]

        except Exception as e:
            print("Error:", e)
            return [{"error": "Failed to make prediction"}]

# Example usage:
if __name__ == "__main__":
    filename = "path/to/your/image.jpg"  # Provide the path to the image file
    pipeline = PredictionPipeline(filename)
    prediction = pipeline.predict()
    print(prediction)
