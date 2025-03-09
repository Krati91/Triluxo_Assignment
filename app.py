import json
from flask import Flask, render_template, \
    request, jsonify

from scripts import retrieve_context_from_vector_store, \
    ask_llm


app = Flask(__name__)


@app.route('/')
def chatbot_menu():
    return render_template('chatbot.html')

@app.route('/respond/', methods=['POST'])
def respond():
    user_input = json.loads(request.data).get('user_input', '')
    context = retrieve_context_from_vector_store(user_input)
    response_text = ask_llm(context, user_input)
    return jsonify({'response': response_text})


if __name__ == '__main__':
    app.run(debug=True)