from flask import Flask, request, jsonify,send_from_directory,render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

from SimilarityServiceV4 import SimilarityServiceV4

app = Flask(__name__)
CORS(app)

base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = f"{base_path}/cars_Dataset_3/"
UPLOAD_FOLDER = os.path.join(base_path, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
model=SimilarityServiceV4("resnet50")

@app.route('/')
def home():
    return render_template('index.html')
@app.route('/similar', methods=['POST'])
def find_similar():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # target = model.get_embedding(filepath)
    top_results = model.find_k_similar_images(filepath)[:5]
    return jsonify([{'path': f"/datasets/train{res[1]}", 'score': float(res[0])} for res in top_results])


@app.route('/datasets/<path:filename>')
def serve_image(filename):
    return send_from_directory(dataset_path, filename)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port='8080')
