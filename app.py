from flask import Flask, render_template, request, jsonify
import os
from script import *
from werkzeug.utils import secure_filename

app = Flask(__name__, static_url_path='', static_folder='static')


UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file:
        filename = secure_filename(file.filename) 
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath) 
        results = query_image_data("static/uploads/" + filename)
        print(results)
        return jsonify({'message': 'File uploaded successfully', 'filepath': results}), 200
    return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=True)
