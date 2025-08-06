from google.genai import types
from google.genai.types import Tool, GenerateContentConfig, GoogleSearch
from app.gemini.gemini_llm import GeminiLLM

class GeminiTools:
    def __init__(self, geminillm: GeminiLLM, model_id="gemini-2.5-flash-preview-05-20"):
        self.geminillm = geminillm
        self.model_id = model_id
        self.google_tool = Tool(google_search=GoogleSearch())
        self.grounding = None

    def stream_thinking_mode(self, prompt: str):
        try:
            stream = self.geminillm.client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    thinking_config=types.ThinkingConfig(include_thoughts=True)
                )
            )

            for chunk in stream:
                if not chunk.candidates:
                    continue

                for part in chunk.candidates[0].content.parts:
                    if hasattr(part, "thought") and part.thought:
                        yield {"text": part.text, "thought": True}
                    elif hasattr(part, "text") and part.text:
                        yield {"text": part.text, "thought": False}

        except Exception as e:
            yield {"text": f"Error during stream_thinking: {str(e)}", "thought": False}

    def stream_google_search(self, prompt: str):
        try:
            stream = self.geminillm.client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[self.google_tool],
                    response_modalities=["TEXT"]
                )
            )

            for chunk in stream:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                self.grounding = getattr(candidate, "grounding_metadata", None)

                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        yield part.text

        except Exception as e:
            yield f"Error during stream_google_search: {str(e)}"

    def stream_code_execution(self, prompt: str):
        try:
            stream = self.geminillm.client.models.generate_content_stream(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(code_execution=types.ToolCodeExecution())],
                    response_modalities=["TEXT"]
                ),
            )

            for chunk in stream:
                if not chunk.candidates:
                    continue

                candidate = chunk.candidates[0]
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        yield part.text
                    elif hasattr(part, "executable_code") and part.executable_code:
                        yield f"\n\n```python\n{part.executable_code.code}\n```\n"
                    elif hasattr(part, "code_execution_result") and part.code_execution_result:
                        yield f"\n\n**Output:**\n```\n{part.code_execution_result.output}\n```\n"

        except Exception as e:
            yield f"Error during stream_code_execution: {str(e)}"

