from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io

app = Flask(__name__, template_folder='.')


model = load_model('potato_classifier_model.h5')


class_labels = [
    'Bacterial spot', 'Early blight', 'Late blight', 'Leaf mold', 'Septoria leaf spot',
    'Two_spotted mite', 'Target spot', 'Yellow leaf curl virus', 'Mosaic virus', 'Healthy'
]
pesticide_suggestion = [
    "Copper-based fungicides such as Bordeaux mixture or copper hydroxide-based products can be effective against bacterial spot."
    "Fungicides containing chlorothalonil, mancozeb, or copper-based products are commonly used for early blight control.",
    "Fungicides containing chlorothalonil, mancozeb, or products with active ingredients like mefenoxam or chloropicrin are effective against late blight.",
    "Copper-based fungicides can help manage leaf mold in tomato plants.",
    "Fungicides containing chlorothalonil or mancozeb can be used to control Septoria leaf spot.",
    "Insecticidal soaps or neem oil can be used to manage two-spotted mites. You can also consider introducing predatory mites as biological control."
    "Fungicides containing chlorothalonil or copper-based products may help control target spot",
    "Managing the whitefly vector is essential for controlling Yellow Leaf Curl Virus. Insecticides like neonicotinoids or pyrethroids can be used to control whiteflies.",
    "There are no direct chemical treatments for mosaic viruses. Prevention through the use of disease-free seeds and controlling aphid populations (which can transmit the virus) is crucial.",
    "To prevent diseases and pests, practice good crop rotation, select disease-resistant varieties, and maintain proper spacing and sanitation in the garden. Additionally, use organic mulches and provide adequate watering and fertilization."
    ]
medicines = [
    "None", "use home remedies like ginger, jaggery, honey and thyme to get relief from the symptoms. Drinking plenty of fluids and taking rest can help you to speed up your recovery. If you notice any symptoms of pneumonia, seek medical help immediately."
    "v","f","w","d","s","a"
]

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    uploaded_file = request.files['image']

    if uploaded_file:

        img_bytes = uploaded_file.read()

        img = image.load_img(io.BytesIO(img_bytes), target_size=(256, 256))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        predicted_label = class_labels[predicted_class]

        return jsonify({'predicted_label': predicted_label},
        {'pesticide_suggestion':pesticide_suggestion})

    return jsonify({'error': 'No image file provided'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
