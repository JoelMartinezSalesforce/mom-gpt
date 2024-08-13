import os
from flask import Flask, request, jsonify

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.llm_plugin_epgt.wrapper.EGPT_wrapper import EGPTQueryHandler
from services.query.main.query_main import Query

app = Flask(__name__)


def functional(user_prompt: str):
    try:


        key = os.getenv('EPGT_API_KEY')
        org_id = os.getenv('ORG_ID')
        if not key or not org_id:
            raise EnvironmentError("API key or Organization ID is not set in environment variables.")

        egpt_model = EGPTQueryHandler(api_key=key, org_id=org_id)
        encoder = JSONEncoder()

        # Place holder collection should replace with the vector ranker
        query = Query("health_data_cons_final", encoder.get_vocab("health_data_cons_final"))
        prompt = user_prompt.lower()
        res = query.ingest(prompt)
        print(f"Query results: {res}")

        return egpt_model.execute_query(prompt, res)
    except Exception as e:
        print(f"Error during query and GPT processing: {e}")
        # Return None or an error message that you can handle in chatbot_response
        return None


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
    app.run(debug=True, port=8080)
