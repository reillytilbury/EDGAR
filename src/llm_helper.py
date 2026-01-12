import time
from typing import Union
# gemini client
from google import genai
from google.genai import types
# anthropic client
import anthropic

def call_llm(
    prompt_text: str,
    model_name: str = "gemini-2.0-flash",
    client: Union[genai.Client, anthropic.Client] = None,
    temperature: float = 1.0,
    thinking_budget: float = 1.0) -> Union[str, None]:
    """
    Send one prompt to the GenAI client and return the text result.
    """
    if model_name[0] == 'g':
        try:
            # create the config for the request (thinking budget for 2.5 flash model)
            if '2.5-flash' in model_name:
                thinking_budget = int(thinking_budget * 24_576)
                config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=5_000, thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget))
            else:
                config = types.GenerateContentConfig(temperature=temperature, max_output_tokens=5_000)
            
            # send the request to the GenAI client
            resp = client.models.generate_content(model=model_name, contents=[prompt_text], config=config)
            return resp.text
        except Exception as e:
            print(f"ERROR (Gemini): {e}")
            # wait a small amount of time before retrying
            time.sleep(5)
            return None
    else:
        try:
            resp = client.messages.create(
                model=model_name,
                max_tokens=5_000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt_text}]
            )
            content = getattr(resp, 'content', [])
            text = getattr(content[0], 'text', '') if content else ""
            return text
        except Exception as e:
            print(f"ERROR (Anthropic): {e}")
            return None
    
async def call_llm_async(
    prompt_text: Union[str, None],
    client: Union[genai.Client, anthropic.Client],
    model_name: str = "gemini-2.0-flash",
    temperature: float = 1.0,
    thinking_budget: float = 1,
    img_bytes: Union[bytes, None] = None
    ) -> Union[str, None]:
    """
    Send one prompt to the GenAI client and return the text result.
    """
    if prompt_text is None:
        return None
    if model_name[0] == 'g':
        try:
            # Create the config for the request (thinking budget for 2.5 flash model)
            if '2.5' in model_name:
                thinking_budget = int(thinking_budget * 24_576) if thinking_budget >= 0 else -1
                config = types.GenerateContentConfig(temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget))
            else:
                config = types.GenerateContentConfig(temperature=temperature)

            # Send the request to the GenAI client
            if img_bytes is not None:
                # If image bytes are provided, include them in the request
                resp = await client.aio.models.generate_content(model=model_name, contents=[prompt_text, types.Part.from_bytes(data=img_bytes, mime_type="image/png")], config=config)
            else:
                # Otherwise, just send the text prompt
                resp = await client.aio.models.generate_content(model=model_name, contents=[prompt_text], config=config)
            
            return resp.text
        except Exception as e:
            print(f"Error in GenAI async call: {e}")
            return None
    else:
        try:
            resp = await client.messages.create(
                model=model_name,
                temperature=temperature,
                max_tokens=5_000,
                messages=[{"role": "user", "content": prompt_text}]
            )
            # Correct way to access Claude's response text
            return resp.content[0].text
        except Exception as e:
            print(f"ERROR (Anthropic): {e}")
            return None
