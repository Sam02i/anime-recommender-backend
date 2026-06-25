from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
# CORS allows your React frontend to talk to this backend
CORS(app) 

@app.route('/api/test', methods=['GET'])
def test():
    # This is the message your frontend is looking for
    return jsonify({"message": "Backend is working!"})

if __name__ == '__main__':
        # Running on port 5000

    app.run(debug=True, port=5000)
    


