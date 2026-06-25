from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import requests
from fuzzywuzzy import process # Import fuzzywuzzy
import os

app = Flask(__name__)
# CORS allows your React frontend to talk to this backend
CORS(app) 

print("loading Ml model & data....")
try:
    with open()
@app.route('/api/test', methods=['GET'])
def test():
    # This is the message your frontend is looking for
    return jsonify({"message": "Backend is working!"})

if __name__ == '__main__':
        # Running on port 5000

    app.run(debug=True, port=5000)
    


