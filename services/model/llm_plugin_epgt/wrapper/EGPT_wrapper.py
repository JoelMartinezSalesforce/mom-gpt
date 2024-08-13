from services.model.llm_plugin_epgt.llm_egpt import EGPT


class Response:
    def __init__(self):
        self.response_json = None

    def text(self):
        return self.response_json.get('text') if self.response_json else None


class Prompt:
    def __init__(self, prompt, options, system=None):
        self.prompt = prompt
        self.options = options
        self.system = system


class EGPTQueryHandler:
    def __init__(self, api_key, org_id):
        self.api_key = api_key
        self.org_id = org_id
        self.egpt_model = EGPT(key=self.api_key)

    def execute_query(self, user_prompt, query):
        org_id = self.org_id

        # Define prompt options
        options = EGPT.Options()
        options.org_id = org_id

        # Create a prompt object
        prompt = Prompt(
            f"Can you explain this information and each field and percentages with the actual values given the "
            f"user prompt and provide at the end an interpretation of the values'{user_prompt}' : {query}",
            options)

        # Create a response object
        response = Response()

        # Execute the model to get a response
        responses = list(self.egpt_model.execute(prompt, stream=False, response=response, conversation=None))

        return responses[0]
