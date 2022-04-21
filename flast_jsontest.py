
from flask import Flask, request, jsonify
from take_input import take_input

app = Flask(__name__)

@app.route('/search', methods=['GET','POST'])
def SearchHandler():
        query_json = request.get_json()
        #query = query_json['query']
        #threshold = float(query_json['threshold'])
        return jsonify(query_json["stuff"])
        #return jsonify(take_input(query=query, distance_threshold=threshold))

app.run(host='0.0.0.0', port=8099)
# app.run()
