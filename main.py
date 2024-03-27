from flask import Flask, request, jsonify
import cv2
import numpy as np
from sklearn.cluster import KMeans
import base64
from io import BytesIO
import logging
from flask_cors import CORS

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

app = Flask(__name__)
CORS(app, origins='*')

logging.basicConfig(level=logging.INFO)

def preprocess_image(base64_image):
    try:
        # Decode base64 image and convert to NumPy array
        image_data = base64.b64decode(base64_image)
        image_np = np.frombuffer(image_data, dtype=np.uint8)
        
        # Check if the image is not empty
        if image_np is None or len(image_np) == 0:
            raise ValueError("Failed to decode the image")
        
        # Decode the image
        image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

        # Check if the image is not empty
        if image is None:
            raise ValueError("Failed to decode the image")

        # Convert image to RGB and normalize pixel values
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb = image_rgb.astype(float) / 255.0

        return image_rgb
    except Exception as e:
        logging.error(f"Error in preprocess_image: {e}")
        return None

def get_dominant_colors(image_rgb, k=15):
    try:
        # Reshape the image to be a list of pixels
        pixels = image_rgb.reshape((-1, 3))

        # Apply K-Means clustering
        kmeans = KMeans(n_clusters=k, n_init=10)
        kmeans.fit(pixels)

        # Get the dominant colors as a list of tuples
        dominant_colors = [tuple(map(int, color * 255)) for color in kmeans.cluster_centers_]

        return dominant_colors
    except Exception as e:
        logging.error(f"Error in get_dominant_colors: {e}")
        return None

@app.route('/analyze', methods=['POST'])
def analyze_image():
    # Get the base64-encoded image from the request
    data = request.get_json()
    base64_image = data.get('base64_image', '')

    # Preprocess the image
    image_rgb = preprocess_image(base64_image)

    if image_rgb is None:
        return jsonify({'error': 'Failed to preprocess image'}), 500

    # Get the dominant colors
    dominant_colors = get_dominant_colors(image_rgb)

    if dominant_colors is None:
        return jsonify({'error': 'Failed to get dominant colors'}), 500

    # Respond with the dominant colors
    response_data = {'dominant_colors': dominant_colors}
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(debug=False)
