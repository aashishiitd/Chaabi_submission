from flask import Flask, request, jsonify
# Assuming Search.py is in the same directory
from Search import get_contextual_answers,get_product

app = Flask(__name__)

@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data['query']
    response = {
        "products": get_product(query),
        "answers": get_contextual_answers(query)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)