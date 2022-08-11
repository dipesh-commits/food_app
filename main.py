from io import BytesIO
from PIL import Image
from segmentation import mask, predict_bbox
from flask import Flask, request, render_template, redirect
import cv2
import glob
import os
import base64, uuid
import numpy as np

import config

from segmentation import predict, mask
from loguru import logger


app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=["GET", "POST"])
def home():
    if request.method == 'POST':
        all_images = glob.glob(os.path.join("static","*.jpg"))
        for f in all_images:
            os.remove(f)
        picture = request.files["file"]
        if picture.filename == '':
                print('No file selected')
                return redirect(request.url)
        if picture and allowed_file(picture.filename):
            org_filename = (picture.filename)
            img = Image.open(picture.stream)
            filename = uuid.uuid4().hex + ".jpg"
            with BytesIO() as buf:
                img.save(buf, 'jpeg')
                img.save(f"static/{filename}")
                image_bytes = buf.getvalue()
                encoded_string = base64.b64encode(image_bytes).decode()

                # predict mask
                output = predict(np.array(img))
                segmented_mask = mask(img, output)
                logger.warning(segmented_mask)
                all_segmentation = []
                for i in segmented_mask:
                    segmented_filename = uuid.uuid4().hex + '.jpg'
                    i.save(f"static/{segmented_filename}")
                    all_segmentation.append(segmented_filename)
                logger.info(f"Segmentation done..")
                detected_img, detected_results = predict_bbox(np.array(img), output)
                detected_filename = uuid.uuid4().hex + '.jpg'
                logger.info(f"Detection done....")
                cv2.imwrite(f"static/{detected_filename}", detected_img)

            return render_template('index.html', img_data = encoded_string, segmented = all_segmentation, detected = detected_filename, results=detected_results, volume=""), 200
    else:
        return render_template('index.html', img_data=""), 200

if __name__ == "__main__":
    # app.run(host='0.0.0.0', port=8080)
    app.run(debug=True)
