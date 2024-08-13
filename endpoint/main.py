import json
import os

import certifi
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from services.model.llm_plugin_epgt.wrapper.EGPT_wrapper import EGPTQueryHandler
from services.query.main.query_main import Query
from slack_sdk import WebClient
from flask_executor import Executor

os.environ['SSL_CERT_FILE'] = certifi.where()
slack_token = os.getenv('SLACK_TOKEN')
VERIFICATION_TOKEN = os.getenv('VERIFICATION_TOKEN')
app = Flask(__name__)

load_dotenv('.env')

executor = Executor(app)

def functional(user_prompt: str):
    try:
        key = os.getenv('EPGT_API_KEY')
        org_id = os.getenv('ORG_ID')
        if not key or not org_id:
            raise EnvironmentError("API key or Organization ID is not set in environment variables.")

        egpt_model = EGPTQueryHandler(api_key=key, org_id=org_id)
        query = Query("health_data_cons_final")
        prompt = user_prompt.lower()
        res = query.ingest(prompt)
        print(f"Query results: {res}")

        return egpt_model.execute_query(prompt, res)
    except Exception as e:
        print(f"Error during query and GPT processing: {e}")
        # Return None or an error message that you can handle in chatbot_response
        return None

@app.route('/', methods=['POST'])
def index():
    data = json.loads(request.data.decode("utf-8"))
    # check the token for all incoming requests
    if data["token"] != VERIFICATION_TOKEN:
        return {"status": 403}
    # confirm the challenge to slack to verify the url
    if "type" in data:
        if data["type"] == "url_verification":
            response = {"challenge": data["challenge"]}
            return jsonify(response)
    # handle incoming mentions - change the "@U078575S5NH" this is the App ID
    if "@U078575S5NH" in data["event"]["text"]:
        # executor will let us send back a 200 right away
        executor.submit(functional,
                        data,
                        data["event"]["text"].
                        replace(f'<@U078575S5NH>', ''))
        return {"status": 200}
    return {"status": 503}


@app.route('/MoMGPT', methods=['POST'])
def chatbot_response():
    if request.is_json:
        data = request.get_json()
        user_message = data.get('message', '')

        if not user_message:
            return jsonify({"error": "Message is empty"}), 400

        gpt_response = functional(user_message)

        if gpt_response:
            return jsonify({"response": gpt_response}), 200
        else:
            return jsonify({"error": "Failed to generate response from GPT model"}), 500
    else:
        return jsonify({"error": "Request must be in JSON format"}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
