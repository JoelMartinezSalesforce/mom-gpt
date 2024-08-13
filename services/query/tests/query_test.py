import os

from services.model.embeddings.corpus.json_encoder import JSONEncoder
from services.model.llm_plugin_epgt.wrapper.EGPT_wrapper import EGPTQueryHandler
from services.query.main.query_main import Query

if __name__ == '__main__':
    key = os.getenv('EPGT_API_KEY')
    org_id = os.getenv('ORG_ID')
    egpt_model = EGPTQueryHandler(api_key=key, org_id=org_id)

    encoder = JSONEncoder()

    vocab = encoder.create_vocab([])

    query = Query("health_data_cons_final", vocabulary=vocab)
    prompt = input("Enter your prompt: ").lower()
    res = query.ingest(prompt)
    print(res)

    print(egpt_model.execute_query(prompt, res))

