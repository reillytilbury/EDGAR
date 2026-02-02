import time
import datetime
import logging
from typing import Union, Optional, List, Any, Callable
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
    
    Supports 4 runtime configurations for different model/mode combinations:
    - large_explore: Large model (2.5) with explore system instruction, higher temperature
    - large_exploit: Large model (2.5) with exploit system instruction, lower temperature
    - small_explore: Small model (2.0) with explore system instruction, higher temperature
    - small_exploit: Small model (2.0) with exploit system instruction, lower temperature
    
    The appropriate config is selected via `ask_island(mode='explore', use_large_model=True)`.
    
    Args:
        client: The Google GenAI client instance.
        get_system_instruction: Callable that takes mode ('explore'/'exploit') and returns system instruction.
        small_model_name: The smaller/faster model (e.g., "gemini-2.0-flash").
        large_model_name: The larger/smarter model (e.g., "gemini-2.5-flash").
        explore_temperature: Temperature for explore mode (higher = more creative).
        exploit_temperature: Temperature for exploit mode (lower = more focused).
        thinking_budget_fraction: Fraction of max thinking budget for 2.5 models (0-1).
    """
    
    def __init__(self, client: genai.Client, 
                 get_system_instruction: Callable[[str], str],
                 small_model_name: str = "gemini-2.0-flash",
                 large_model_name: str = "gemini-2.5-flash",
                 explore_temperature: float = 1.5,
                 exploit_temperature: float = 0.7,
                 thinking_budget_fraction: float = 1.0):
        self.client = client
        self.get_system_instruction = get_system_instruction
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name
        self.explore_temperature = explore_temperature
        self.exploit_temperature = exploit_temperature
        self.thinking_budget_fraction = thinking_budget_fraction
        
        # Dictionary to store chat sessions: { island_id: chat_object }
        self.islands: dict[int, genai.chats.AsyncChats] = {}
        
        # Build the 4 configs
        self._build_configs()
        
        # Track the current mode for logging
        self.current_mode = 'explore'
        self.current_use_large = True
    
    def _build_configs(self) -> None:
        """Build the 4 GenerateContentConfig objects for each model/mode combination."""
        max_thinking_budget = 24_576
        thinking_budget = int(self.thinking_budget_fraction * max_thinking_budget)
        
        # Get system instructions for both modes
        explore_instruction = self.get_system_instruction('explore')
        exploit_instruction = self.get_system_instruction('exploit')
        
        # Large model configs (with thinking budget for 2.5)
        self.large_explore_config = types.GenerateContentConfig(
            temperature=self.explore_temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            system_instruction=explore_instruction
        )
        self.large_exploit_config = types.GenerateContentConfig(
            temperature=self.exploit_temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
            system_instruction=exploit_instruction
        )
        
        # Small model configs (no thinking budget)
        self.small_explore_config = types.GenerateContentConfig(
            temperature=self.explore_temperature,
            system_instruction=explore_instruction
        )
        self.small_exploit_config = types.GenerateContentConfig(
            temperature=self.exploit_temperature,
            system_instruction=exploit_instruction
        )
        
        # Store for easy access
        self.configs = {
            ('large', 'explore'): self.large_explore_config,
            ('large', 'exploit'): self.large_exploit_config,
            ('small', 'explore'): self.small_explore_config,
            ('small', 'exploit'): self.small_exploit_config,
        }
        
        # Store system instructions for logging
        self.explore_instruction = explore_instruction
        self.exploit_instruction = exploit_instruction
    
    def get_config(self, mode: str = 'explore', use_large_model: bool = True) -> types.GenerateContentConfig:
        """Get the appropriate config for the given mode and model size."""
        size = 'large' if use_large_model else 'small'
        return self.configs[(size, mode)]
    
    def get_model_name(self, use_large_model: bool = True) -> str:
        """Get the model name for the given size."""
        return self.large_model_name if use_large_model else self.small_model_name
    
    def log_configuration(self) -> None:
        """
        Log the IslandChatManager configuration to the current logging handler.
        
        Call this method AFTER the logging file handler is set up to ensure
        the system instruction is captured in the log file.
        """
        logging.info("="*80)
        logging.info("ISLAND CHAT MANAGER CONFIGURATION")
        logging.info("="*80)
        logging.info(f"Small Model: {self.small_model_name}")
        logging.info(f"Large Model: {self.large_model_name}")
        logging.info(f"Explore Temperature: {self.explore_temperature}")
        logging.info(f"Exploit Temperature: {self.exploit_temperature}")
        logging.info(f"Thinking Budget Fraction: {self.thinking_budget_fraction}")
        logging.info(f"Active islands: {list(self.islands.keys()) if self.islands else 'None yet'}")
        logging.info("")
        logging.info("EXPLORE MODE SYSTEM INSTRUCTION:")
        logging.info("-"*40)
        for line in self.explore_instruction.split('\n'):
            logging.info(line)
        logging.info("-"*40)
        logging.info("")
        logging.info("EXPLOIT MODE SYSTEM INSTRUCTION:")
        logging.info("-"*40)
        for line in self.exploit_instruction.split('\n'):
            logging.info(line)
        logging.info("-"*40)
        logging.info("="*80)

    def _create_island_chat(self, island_id: int, mode: str = 'explore', 
                            use_large_model: bool = True) -> genai.chats.AsyncChats:
        """
        Create a new chat session for an island.
        
        Args:
            island_id: The island identifier.
            mode: 'explore' or 'exploit' - determines system instruction.
            use_large_model: If True, use the large model; otherwise use small model.
            
        Returns:
            A new async chat object.
        """
        config = self.get_config(mode, use_large_model)
        model_name = self.get_model_name(use_large_model)
        
        chat = self.client.aio.chats.create(
            model=model_name,
            config=config,
            history=[],
        )
        
        logging.info(f"Created new chat session for Island {island_id} "
                    f"(model={model_name}, mode={mode})")
        return chat

    async def get_or_create_island(self, island_id: int, mode: str = 'explore',
                                   use_large_model: bool = True) -> genai.chats.AsyncChats:
        """
        Get the chat session for an island, creating one if it doesn't exist.
        
        Args:
            island_id: The island identifier.
            mode: 'explore' or 'exploit' for new chat creation.
            use_large_model: Model size for new chat creation.
            
        Returns:
            The chat object for this island.
        """
        if island_id not in self.islands:
            self.islands[island_id] = self._create_island_chat(island_id, mode, use_large_model)
        return self.islands[island_id]

    async def ask_island(self, island_id: int, prompt: str, 
                         mode: str = 'explore',
                         use_large_model: bool = True,
                         png_img: Optional[bytes] = None) -> str:
        """
        Send a message to an island's chat and get a response.
        
        The message is automatically appended to the island's chat history,
        allowing the LLM to see and learn from previous interactions.
        
        Uses override config to dynamically switch between model/mode combinations
        without recreating the chat session.
        
        Args:
            island_id: The island identifier.
            prompt: The prompt text to send.
            mode: 'explore' or 'exploit' - affects system instruction and temperature.
            use_large_model: If True, use large model config; otherwise small model.
            png_img: Optional PNG image bytes to include with the prompt.
            
        Returns:
            The LLM's response text, or empty string on error.
        """
        chat = await self.get_or_create_island(island_id, mode, use_large_model)
        
        # Get the appropriate override config for this request
        override_config = self.get_config(mode, use_large_model)
        
        # Track for logging
        self.current_mode = mode
        self.current_use_large = use_large_model
        
        try:
            if png_img is not None:
                # Create the parts for the multi-modal message
                message_parts = [
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(data=png_img, mime_type="image/png")
                ]
            else:
                message_parts = [types.Part.from_text(text=prompt)]
            
            # Use override config to apply the appropriate settings
            response = await chat.send_message(message_parts, config=override_config)
            return response.text
        except Exception as e:
            model_name = self.get_model_name(use_large_model)
            print(f"Error sending message to island {island_id} chat "
                  f"(model={model_name}, mode={mode}): {e}")
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
                chat = self.islands[island_id]
                # get_history() may or may not be async depending on SDK version
                history = chat.get_history()
                # If it's a coroutine, await it
                if hasattr(history, '__await__'):
                    history = await history
                return list(history) if history else []
            else:
                print(f"No chat found for island_id {island_id}")
                return []
        except Exception as e:
            print(f"Error retrieving island history for island_id={island_id}: {e}")
            return []
    
    def get_n_islands(self) -> int:
        """Return the number of active island chats."""
        return len(self.islands)
    
    def reset_island(self, island_id: int, mode: str = 'explore', 
                     use_large_model: bool = True) -> None:
        """
        Reset an island's chat history by creating a fresh chat.
        
        Useful if the chat history grows too large or needs to be cleared.
        
        Args:
            island_id: The island identifier to reset.
            mode: 'explore' or 'exploit' for the new chat.
            use_large_model: Model size for the new chat.
        """
        if island_id in self.islands:
            self.islands[island_id] = self._create_island_chat(island_id, mode, use_large_model)
            logging.info(f"Reset chat session for Island {island_id}")
    
    def reset_all_islands(self, mode: str = 'explore', use_large_model: bool = True) -> None:
        """Reset all island chat histories."""
        for island_id in list(self.islands.keys()):
            self.reset_island(island_id, mode, use_large_model)

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
