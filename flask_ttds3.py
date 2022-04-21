from flask import Flask, request, jsonify
from take_input import take_input
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
model = SentenceTransformer('snt_tsfm_model/')
model.encode('random query')

@app.route('/')
def queryHandler():
	query = request.args.get('query')
	query_str = str(query).replace('_',' ')
	query_encoded = model.encode(query_str)
	return jsonify(take_input(query_encoded))


app.run(host='0.0.0.0', port=8090)