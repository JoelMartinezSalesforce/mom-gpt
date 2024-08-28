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
            f"Can you explain the provided information, detailing each field and the percentages along with their "
            f"actual values, based on the user prompt? At the end, please include an interpretation of the values. "
            f"This is the user prompt: '{user_prompt}'. This is the information you need to use to respond: '"
            f"{query}'. Always respond in the first person and format the information the best you can.",
            options)

        # Create a response object
        response = Response()

        # Execute the model to get a response
        responses = list(self.egpt_model.execute(prompt, stream=False, response=response, conversation=None))

        return responses[0]
