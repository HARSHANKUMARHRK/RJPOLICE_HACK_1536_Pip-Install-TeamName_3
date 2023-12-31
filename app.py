from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from pymongo import MongoClient
import urllib.parse

app = Flask(__name__)
app.secret_key = 'abc'  # Add a secret key for sessions

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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
