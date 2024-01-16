# Import the Flask class from the flask module
from flask import Flask

# Create an instance of the Flask class
app = Flask(__name__)

# Define a route for the root URL ("/")
@app.route('/')
def hello_world():
    return 'Hello, World!'

# Run the application if the script is executed
if __name__ == '__main__':
    # Run the app on localhost (127.0.0.1) and port 5000
    app.run(debug=True)
