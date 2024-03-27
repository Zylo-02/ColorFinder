from flask import Flask, request, jsonify
from io import BytesIO
from sklearn.cluster import KMeans
from PIL import Image

app = Flask(__name__)

# ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.lower().endswith(('png', 'jpg', 'jpeg'))

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(int(rgb[0]), int(rgb[1]), int(rgb[2]))



@app.route('/dominant_colors', methods=['POST'])
def get_dominant_colors():
    if request.method == 'POST':
        # Check for uploaded file
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Print filename and content type
        print("Filename:", file.filename)
        print("Content type:", file.content_type)

        if file and allowed_file(file.filename):
            # Read image data
            image_data = BytesIO(file.read())
            image = Image.open(image_data)

            # Convert image to suitable format (e.g., RGB)
            image_data = list(image.getdata())

            # K-means clustering
            kmeans = KMeans(n_clusters=10, random_state=0, n_init=10).fit(image_data)  # Adjust k for desired number of colors

            # Extract dominant colors (centroids)
            dominant_colors = kmeans.cluster_centers_.tolist()

            # Convert RGB values to hex format
            dominant_colors_hex = ['"{}"'.format(rgb_to_hex(color)) for color in dominant_colors]

            return jsonify({'dominant_colors': dominant_colors_hex})

        else:
            return jsonify({'error': 'Unsupported file format'}), 400

    else:
        return jsonify({'error': 'Method not allowed'}), 405




if __name__ == '__main__':
    app.run(debug=True)
