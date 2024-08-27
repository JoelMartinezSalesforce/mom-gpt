import llm
from typing import Optional
import requests
import urllib3

urllib3.disable_warnings()

@llm.hookimpl
def register_models(register):
    register(EGPT())

def generate_chat_response(messages, key, gateway_url, model, org_id, client_feature_id, llm_provider):
    # TODO: need to add CA cert validation for the falcon cert chain instead of ignoring
    resp = requests.post(f"{gateway_url}/v1.0/chat/generations", json={
        "model": model,
        "messages": messages,
        "generation_settings": {
            "num_generations": 1,
            "max_tokens": 3000,
        },
    }, headers={
        "x-llm-provider": llm_provider,
        "x-client-feature-id": client_feature_id,
        "authorization": f"API_KEY {key}",
        "x-sfdc-core-tenant-id": f"core/prod1/{org_id}",
        'x-org-id': org_id,
    }, verify=False)
    resp.raise_for_status()
    return resp.json()


def generate_response(prompt, key, gateway_url, model, org_id, client_feature_id, llm_provider):
    # TODO: need to add CA cert validation for the falcon cert chain instead of ignoring
    resp = requests.post(f"{gateway_url}/v1.0/generations", json={
        "model": model,
        "prompt": prompt
    }, headers={
        "x-llm-provider": llm_provider,
        "x-client-feature-id": client_feature_id,
        "authorization": f"API_KEY {key}",
        "x-sfdc-core-tenant-id": f"core/prod1/{org_id}",
        'x-org-id': org_id,
    }, verify=False)
    resp.raise_for_status()
    return resp.json()


class EGPT(llm.Model):
    model_id = "egpt"
    needs_key = "egpt"
    key_env_var = "EGPT_API_KEY"
    can_stream: bool = False

    class Options(llm.Options):
        model: Optional[str] = "gpt-3.5-turbo"
        client_feature_id: Optional[str] = "ECOMQLabs"
        llm_provider: Optional[str] = "OpenAI"
        gateway_url: Optional[str] = "https://bot-svc-llm.sfproxy.einstein.perf2-uswest2.aws.sfdc.cl"
        org_id: str = None

    def __init__(self, key=None):
        self.key = key

    def execute(self, prompt, stream, response, conversation):
        assert self.key is not None, "You must provide an API key for EGPT"
        model = prompt.options.model
        org_id = prompt.options.org_id

        assert org_id is not None, "You must provide an Core Org ID for EGPT: -o org_id abcd1234"

        messages = []
        current_system = None
        if conversation is not None:
            for prev_response in conversation.responses:
                if (
                        prev_response.prompt.system
                        and prev_response.prompt.system != current_system
                ):
                    messages.append(
                        {"role": "system", "content": prev_response.prompt.system}
                    )
                    current_system = prev_response.prompt.system
                messages.append(
                    {"role": "user", "content": prev_response.prompt.prompt}
                )
                messages.append({"role": "assistant", "content": prev_response.text()})

        # use a chat session vs use a generation
        use_chat = len(messages) > 0

        _prompt = prompt.prompt
        if use_chat:
            if prompt.system and prompt.system != current_system:
                messages.append({"role": "system", "content": prompt.system})
            messages.append({"role": "user", "content": prompt.prompt})
        elif prompt.system:
            _prompt = f"{prompt.system}\n{prompt.prompt}"

        if use_chat:
            data = generate_chat_response(messages, key=self.key, gateway_url=prompt.options.gateway_url,
                                          model=model,
                                          org_id=org_id,
                                          client_feature_id=prompt.options.client_feature_id,
                                          llm_provider=prompt.options.llm_provider)
            response.response_json = data
            yield data.get('generation_details').get('generations')[0].get('content')
        else:
            data = generate_response(_prompt, key=self.key, gateway_url=prompt.options.gateway_url,
                                     model=model,
                                     org_id=org_id,
                                     client_feature_id=prompt.options.client_feature_id,
                                     llm_provider=prompt.options.llm_provider)
            response.response_json = data
            yield data.get('generations')[0].get('text')
