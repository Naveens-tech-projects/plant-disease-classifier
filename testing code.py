import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

model = load_model('potato_classifier_model.h5')


class_labels = [
    'Bacterial spot', 'Early blight', 'Late blight', 'Leaf mold', 'Septoria leaf spot',
    'Two_spotted mite', 'Target spot', 'Yellow leaf curl virus', 'Mossaic virus', 'Healthy'
]


test_image_path = r'E:\New Plant Diseases Dataset(Augmented)\valid\Tomato___healthy\1b477a6a-aa09-4c12-ad17-fbbc08cd76da___GH_HL Leaf 386.JPG'
img = image.load_img(test_image_path, target_size=(256, 256))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

predictions = model.predict(img_array)
predicted_class = np.argmax(predictions)


print("The given leaf image comes under the category:", class_labels[predicted_class])


