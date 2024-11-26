from flask import Flask, request, render_template, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# Initialize Flask app
app = Flask(__name__)

# Set the folder to save uploaded images
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the trained model
model = load_model('crater_detection_model.keras')

# Define image size
IMAGE_SIZE = (128, 128)

def classify_image(image_path, model):

    img = load_img(image_path, target_size=IMAGE_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    return 'No Craters Detected' if prediction[0][0] > 0.5 else 'Craters Detected'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        
        # Save the uploaded file
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)
        
        # Make a prediction using the model
        result = classify_image(file_path, model)
        
        # Pass the file path of the uploaded image to the template for display
        image_url = url_for('static', filename=f'uploads/{file.filename}')
        
        return render_template('index.html', result=result, image_url=image_url)
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)  # Ensure upload folder exists
    app.run(debug=True)
