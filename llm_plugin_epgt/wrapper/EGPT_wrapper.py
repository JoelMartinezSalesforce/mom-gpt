from services.model.llm_plugin_epgt.llm_egpt import EGPT


class Response:
    """
    A class to represent the response from the EGPT model.

    Attributes:
        response_json (dict): A dictionary to hold the JSON response from the model.

    Methods:
        text():
            Extracts and returns the 'text' field from the JSON response.
    """
    def __init__(self):
        """
        Initializes the Response object with a placeholder for the JSON response.
        """
        self.response_json = None

    def text(self):
        """
        Retrieves the 'text' content from the JSON response.

        Returns:
            str or None: The text content if available, otherwise None.
        """
        return self.response_json.get('text') if self.response_json else None


class Prompt:
    def __init__(self, prompt, options, system=None):
        """
        A class to represent the prompt used to query the EGPT model.

        Attributes:
            prompt (str): The text of the prompt to be sent to the model.
            options (EGPT.Options): Configuration options for the model's response.
            system (str, optional): System information that might affect the prompt's behavior. Defaults to None.

        Methods:
            None
        """
        self.prompt = prompt
        self.options = options
        self.system = system


class EGPTQueryHandler:
    """
    A handler class to manage the process of querying the EGPT model.

    Attributes:
        api_key (str): The API key for accessing the EGPT model.
        org_id (str): The organization ID for contextualizing the query.
        egpt_model (EGPT): An instance of the EGPT model initialized with the provided API key.

    Methods:
        execute_query(user_prompt, query):
            Executes a query against the EGPT model and returns the model's response.
    """
    def __init__(self, api_key, org_id):
        """
        Initializes the EGPTQueryHandler with API credentials and sets up the EGPT model.

        Args:
            api_key (str): The API key required for authenticating with the EGPT model.
            org_id (str): The organization ID used to tailor the query context.
        """
        self.api_key = api_key
        self.org_id = org_id
        self.egpt_model = EGPT(key=self.api_key)

    def execute_query(self, user_prompt, query):
        """
        Executes a query to the EGPT model using the provided user prompt and additional query information.

        Args:
            user_prompt (str): The user-provided prompt or question.
            query (str): The additional information or data that the model should use to formulate its response.

        Returns:
            str: The text response from the EGPT model.
        """
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
