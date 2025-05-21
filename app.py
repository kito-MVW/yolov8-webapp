import os
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import numpy as np

app = Flask(__name__)

# โหลด YOLOv8 โมเดล
model = YOLO("best.pt")

# โฟลเดอร์สำหรับเก็บภาพที่อัปโหลดและผลลัพธ์
UPLOAD_FOLDER = os.path.join('static', 'uploads')
PREDICTION_FOLDER = os.path.join('static', 'predictions')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICTION_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            # เซฟไฟล์ภาพที่อัปโหลด
            filename = secure_filename(file.filename)
            upload_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(upload_path)

            # ทำการทำนายด้วย YOLOv8
            results = model.predict(source=upload_path, save=False, conf=0.4)

            # วาดกรอบบนภาพ
            for r in results:
                im_array = r.plot()  # image as numpy array with boxes
                im = Image.fromarray(im_array)
                pred_path = os.path.join(PREDICTION_FOLDER, filename)
                im.save(pred_path)

                # สร้างผลลัพธ์สำหรับแสดง
                prediction = []
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    cls_name = model.names[cls_id]
                    conf = float(box.conf[0]) * 100
                    prediction.append(f"{cls_name} ({conf:.1f}%)")

            # ส่ง path ไปยัง template
            image_path = url_for('static', filename=f'predictions/{filename}')

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
