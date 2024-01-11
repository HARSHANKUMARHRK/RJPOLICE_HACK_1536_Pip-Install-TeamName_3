from flask import Flask, render_template, request, redirect, url_for, session,Response
from flask_session import Session
from pymongo import MongoClient
import urllib.parse
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch
import os
import cv2
import requests

app = Flask(__name__)

#static/output
def get_ip_address():
    url = 'https://api.ipify.org'
    response = requests.get(url)
    ip_address = response.text
    return ip_address

def extract_frames(video_path, output_folder):
    video_reader = cv2.VideoCapture(video_path)
    frame_count = 0
    while video_reader.isOpened():
        ret, frame = video_reader.read()
        if not ret:
            break
        frame_path = f"{output_folder}/frame_{frame_count}.jpg"
        cv2.imwrite(frame_path, frame)

        frame_count += 1
        print(frame_count)
    video_reader.release()
# input_video_path = "intro.mp4"
output_frames_folder = "static/output"
# extract_frames(input_video_path, output_frames_folder)

def video_processing():
    cap = cv2.VideoCapture(0)
    ret, prev_frame = cap.read()
    c = 0
    displacement_threshold = 20

    while True:
        ret, next_frame = cap.read()

        if not ret:
            break
        frame_diff = cv2.absdiff(prev_frame, next_frame)
        frame_diff_gray = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2RGB)
        mean_diff = cv2.mean(frame_diff_gray)[0]

        if mean_diff > displacement_threshold:
            c += 1
            ip=get_ip_address()
            loc=get_location_from_ip(ip)
            print(c)
            print(loc)
            print("camera displaced")

        ret, buffer = cv2.imencode('.jpg', frame_diff)
        frame_diff = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_diff + b'\r\n')

        prev_frame = next_frame.copy()

def get_location_from_ip(ip_address):
    access_token = 'ca2235bc0acff2'  # Replace with your actual access token from ipinfo.io
    url = f'https://ipinfo.io/{ip_address}?token={access_token}'
    
    response = requests.get(url)
    data = response.json()

    city = data.get('city')
    region = data.get('region')
    country = data.get('country')
    
    location = f"{city}, {region}, {country}"
    return location

processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
app.secret_key = 'abc'
folder_path = "static/output"
mongodb_uri = "mongodb+srv://harshan:" + urllib.parse.quote("Harshan@1803") + "@cluster0.ixfzmsm.mongodb.net/?retryWrites=true&w=majority"
database_name = 'credentials_db'
client = MongoClient(mongodb_uri)
db = client[database_name]
users_collection = db['users']

@app.route('/')
def start():
    return redirect(url_for('login'))

@app.route('/login', methods=['POST', 'GET'])
def login():
    if request.method == 'POST':
        user_auth = request.form['nm']
        pwd_auth = request.form['pwd']
        user_data = users_collection.find_one({'email': user_auth})

        if user_data and str(user_data['password']) == pwd_auth:
            session['user'] = str(user_data['_id'])  
            return redirect(url_for('home'))
    
    return render_template('login.html')

@app.route('/home')
def home():
    return render_template('home.html',user=session["user"])

@app.route('/frames', methods=['POST', 'GET'])
def frames():
    detections = []

    if request.method == 'POST':
        video_file = request.files['video']
        prompt = request.form['prompt']

        video_path = 'uploaded_video.mp4' 
        video_file.save(video_path)
        # extract_frames(video_path, output_frames_folder)

        for filename in os.listdir(folder_path):
            if filename.endswith(".jpg"): 
                image_path = os.path.join(folder_path, filename)
                image = Image.open(image_path)
                inputs = processor(images=image, return_tensors="pt")
                outputs = model(**inputs)

                target_sizes = torch.tensor([image.size[::-1]])
                results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

                for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
                    if score >= 0.5 and model.config.id2label[label.item()] == prompt:
                        box = [round(coord, 2) for coord in box.tolist()]
                        detection_info = f"Detected {model.config.id2label[label.item()]} with confidence " \
                                        f"{round(score.item(), 3)} at location {box}"
                        detections.append({"image_path": image_path, "info": detection_info})


#output/frame_59.jpg
    return render_template('frame.html', detections=detections)

@app.route('/displacement')
def displacement():
    return render_template('displacement.html')

@app.route('/video_feed')
def video_feed():
    return Response(video_processing(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)

