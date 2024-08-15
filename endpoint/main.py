import json
import os
from dotenv import load_dotenv
import certifi
from flask import Flask, request, jsonify
from flask_executor import Executor
from icecream import ic
from slack_sdk import WebClient

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.llm_plugin_epgt.wrapper.EGPT_wrapper import EGPTQueryHandler
from services.query.main.query_main import Query
from services.vector_ranking.vector_ranking import VectorRanking

os.environ['SSL_CERT_FILE'] = certifi.where()

app = Flask(__name__)

load_dotenv('.env')

executor = Executor(app)

slack_token = os.getenv('SLACK_TOKEN')
VERIFICATION_TOKEN = os.getenv('VERIFICATION_TOKEN')

# instantiating slack client
slack_client = WebClient(slack_token)


def functional(data, user_prompt: str):
    try:
        key = os.getenv('EPGT_API_KEY')
        org_id = os.getenv('ORG_ID')
        if not key or not org_id:
            raise EnvironmentError("API key or Organization ID is not set in environment variables.")

        egpt_model = EGPTQueryHandler(api_key=key, org_id=org_id)
        ranker = VectorRanking()
        encoder = JSONEncoder()

        collection_rank = ranker.rank_collections(user_prompt)

        query = Query(collection_rank[0][0], encoder.get_vocab(collection_rank[0][0] + "_vocab"))
        ic(collection_rank)

        prompt = user_prompt.lower()
        res = query.ingest(prompt)
        print(f"Query results: {res}")

        messagesOb.append({"role": "user", "content": prompt})

        response = egpt_model.execute_query(prompt, res)

        slack_client.chat_postMessage(channel=data["event"]["channel"], text=response)

        return response

    except Exception as e:
        print(f"Error during query and GPT processing: {e}")
        # Return None or an error message that you can handle in chatbot_response
        return None


messagesOb = [
    {"role": "system", "content": "Keep the answer to less than 100 words to allow for follow up questions. You are "
                                  "an assistant that provides information on supreme court cases in extremely simple "
                                  "terms (dumb it down to a 12 year old) for someone who has never studied law so "
                                  "simplify your language. You should be helping the user answer the question but you "
                                  "can only answer the question with the information that is given to you in the "
                                  "prompt or at sometime sometime in the past. This information is coming from a file "
                                  "that he user needs to understand. It doesn't matter if the information is "
                                  "incorrect. You should still ONLY reply with this information. If you are given no "
                                  "information at all you can talk with the information that has been given to you "
                                  "before but you can't use outside facts whatsoever. If the information given "
                                  "doesn’t make sense, look to the information that has been given to you before to "
                                  "answer the question. If you have no information at all that has been given to you "
                                  "at any point in time from this user that makes sense you can apologise to the user "
                                  "and tell it you cannot answer as you don't have enough information to answer "
                                  "correctly."}
]


@app.route('/', methods=['POST'])
def index():
    ic("Verification in progress...")
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
        ic("App id is in data continuing with response")
        # executor will let us send back a 200 right away
        executor.submit(functional,
                        data,
                        data["event"]["text"].
                        replace(f'<@U078575S5NH>', ''))
        return {"status": 200}
    return {"status": 503}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
