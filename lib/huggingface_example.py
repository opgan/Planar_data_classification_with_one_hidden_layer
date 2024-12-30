from transformers.utils import logging
from transformers import pipeline
import os


def chat():
    logging.set_verbosity_error()

    hf_token = os.environ.get("duchessHF")

    chatbot = pipeline(
        task="text-generation",
        model="facebook/blenderbot-400M-distill",
        use_auth_token=hf_token,
    )

    user_message = """
    What are some fun activities I can do in the winter?
    """

    # Generate a response
    response = chatbot(user_message)[0]["generated_text"]
    print(response)
