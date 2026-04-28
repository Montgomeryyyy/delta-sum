import os

from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions

RANDOM_SEED = 42

client = genai.Client(
    vertexai=True,
    project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
    location="us-central1",
    http_options=HttpOptions(api_version="v1"),
)

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Tell me a joke",
    config=GenerateContentConfig(
        response_logprobs=True,
        logprobs=3,
        seed=RANDOM_SEED,
        # thinking_level = ThinkingConfig(thinking_level=ThinkingLevel.LOW)
    ),
)
print(response)

client.close()
