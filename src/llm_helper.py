import time
import datetime
import logging
from typing import Union, Optional, List, Any
# gemini client
from google import genai
from google.genai import types, chats
# anthropic client
import anthropic

class IslandChatManager:
    """
    Manages persistent chat sessions for each island in the genetic algorithm.
    
    Each island maintains its own chat history, allowing the LLM to learn from
    the context of previous generations within that island's evolutionary lineage.
    
    Args:
        client: The Google GenAI client instance.
        system_instruction: The system instruction to use for all island chats.
                           This should contain the static guidelines (code style,
                           function signatures, etc.) that don't change per query.
        model_name: The default model to use for chat sessions.
        temperature: Default temperature for generation.
    """
    
    def __init__(self, client: genai.Client, system_instruction: str, 
                 model_name: str = "gemini-2.0-flash", temperature: float = 1.0):
        self.client = client
        self.model_name = model_name
        self.system_instruction = system_instruction
        self.temperature = temperature
        # Dictionary to store chat sessions: { island_id: chat_object }
        self.islands: dict[int, genai.chats.AsyncChats] = {}
    
    def log_configuration(self) -> None:
        """
        Log the IslandChatManager configuration to the current logging handler.
        
        Call this method AFTER the logging file handler is set up to ensure
        the system instruction is captured in the log file.
        """
        logging.info("="*80)
        logging.info("ISLAND CHAT MANAGER CONFIGURATION")
        logging.info("="*80)
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Temperature: {self.temperature}")
        logging.info(f"Active islands: {list(self.islands.keys()) if self.islands else 'None yet'}")
        logging.info("")
        logging.info("SYSTEM INSTRUCTION (sent to all island chats):")
        logging.info("-"*40)
        for line in self.system_instruction.split('\n'):
            logging.info(line)
        logging.info("-"*40)
        logging.info("="*80)
    
    def _create_chat_config(self, temperature: Optional[float] = None, system_instruction: Optional[str] = None) -> types.GenerateContentConfig:
        """Create a GenerateContentConfig for chat creation."""
        temp = temperature if temperature is not None else self.temperature
        system_instruction = system_instruction if system_instruction is not None else self.system_instruction
        
        if '2.5' in self.model_name:
            # Default thinking budget for 2.5 models
            thinking_budget = int(1.0 * 24_576)            
            config = types.GenerateContentConfig(
                temperature=temp,
                thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
                system_instruction=system_instruction
            )
        else:
            config = types.GenerateContentConfig(temperature=temp, 
                                                 system_instruction=system_instruction)
        
        return config

    def _create_island_chat(self, island_id: int) -> genai.chats.AsyncChats:
        """
        Create a new chat session for an island.
        
        Args:
            island_id: The island identifier.
            
        Returns:
            A new async chat object.
        """
        config = self._create_chat_config(system_instruction=self.system_instruction)
        
        chat = self.client.aio.chats.create(
            model=self.model_name,
            config=config,
            history=[],
        )
        
        logging.info(f"Created new chat session for Island {island_id} (model={self.model_name})")
        return chat

    async def get_or_create_island(self, island_id: int) -> genai.chats.AsyncChats:
        """
        Get the chat session for an island, creating one if it doesn't exist.
        
        Args:
            island_id: The island identifier.
            
        Returns:
            The chat object for this island.
        """
        if island_id not in self.islands:
            self.islands[island_id] = self._create_island_chat(island_id)
        return self.islands[island_id]

    async def ask_island(self, island_id: int, prompt: str, 
                         png_img: Optional[bytes] = None) -> str:
        """
        Send a message to an island's chat and get a response.
        
        The message is automatically appended to the island's chat history,
        allowing the LLM to see and learn from previous interactions.
        
        Args:
            island_id: The island identifier.
            prompt: The prompt text to send.
            png_img: Optional PNG image bytes to include with the prompt.
            
        Returns:
            The LLM's response text, or empty string on error.
        """
        chat = await self.get_or_create_island(island_id)
        
        try:
            if png_img is not None:
                # Create the parts for the multi-modal message
                message_parts = [
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=png_img, mime_type="image/png")
                ]
            else:
                message_parts = [types.Part.from_text(text=prompt)]
            response = await chat.send_message(message_parts)
            return response.text
        except Exception as e:
            print(f"Error sending message to island {island_id} chat: {e}")
            return ""

    async def get_island_history(self, island_id: int) -> list:
        """
        Retrieve the chat history for an island.
        
        Args:
            island_id: The island identifier.
            
        Returns:
            The chat history list, or empty list if not found/error.
        """
        try:
            if island_id in self.islands:
                current_history = await self.islands[island_id].get_history()
                return current_history
            else:
                print(f"No chat found for island_id {island_id}")
                return []
        except Exception as e:
            print(f"Error retrieving island history for island_id={island_id}: {e}")
            return []
    
    def get_n_islands(self) -> int:
        """Return the number of active island chats."""
        return len(self.islands)
    
    def reset_island(self, island_id: int) -> None:
        """
        Reset an island's chat history by creating a fresh chat.
        
        Useful if the chat history grows too large or needs to be cleared.
        
        Args:
            island_id: The island identifier to reset.
        """
        if island_id in self.islands:
            self.islands[island_id] = self._create_island_chat(island_id)
    
    def reset_all_islands(self) -> None:
        """Reset all island chat histories."""
        for island_id in list(self.islands.keys()):
            self.reset_island(island_id)

async def dnu_switch_gemini_model(
    chat: genai.chats.AsyncChats,
    new_model_name: str,
    temperature: float = 1.0,
    thinking_budget: float = 1.0,
    ) -> None:
    """
    Switch the model of an existing Gemini chat client. Do not use this until we figure out how expensive it is to recreate chats with the same history.
    Args:
        chat: An instance of genai.aio.chats.Chat.
        new_model_name: Name of the new Gemini model to switch to.
    """
    # Create the config for the request (thinking budget for 2.5 flash model)
    if '2.5' in new_model_name:
        thinking_budget = int(thinking_budget * 24_576) if thinking_budget >= 0 else -1
        config = types.GenerateContentConfig(temperature=temperature, thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget))
    else:
        config = types.GenerateContentConfig(temperature=temperature)
    chat.model = new_model_name
    chat.config = config

    current_history = await chat.get_history()

    # create a new chat with a different model using the same history
    new_chat = chat.client.aio.chats.create(
        model=new_model_name,
        config=config,
        history=current_history,
    )
    return new_chat

# ---------------------------------------------------------------
# legacy functions 
# ---------------------------------------------------------------
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
