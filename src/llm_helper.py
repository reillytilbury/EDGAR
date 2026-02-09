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
    Manages persistent chat sessions for each (island, batch) pair in the genetic algorithm.
    
    Each (island, batch) combination maintains its own chat history, allowing the LLM 
    to learn from the context of previous generations within that specific lineage.
    This provides diversity across batches while maintaining history within each batch.
    
    With n_islands=8 and batch_size=6, this creates 48 independent chat sessions.
    This is actually MORE cost-effective than per-island chats because each chat
    accumulates history 6x slower (1 message per iteration vs 6).
    
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
        chat_token_limit: Max tokens per chat before auto-summarize and reset. 0 = unlimited.
        batch_size: Number of batches per island (for logging aggregation).
    """
    
    def __init__(self, client: genai.Client, 
                 get_system_instruction: Callable[[str], str],
                 small_model_name: str = "gemini-2.0-flash",
                 large_model_name: str = "gemini-2.5-flash",
                 explore_temperature: float = 1.5,
                 exploit_temperature: float = 0.7,
                 thinking_budget_fraction: float = 1.0,
                 chat_token_limit: int = 50000,
                 batch_size: int = 6):
        self.client = client
        self.get_system_instruction = get_system_instruction
        self.small_model_name = small_model_name
        self.large_model_name = large_model_name
        self.explore_temperature = explore_temperature
        self.exploit_temperature = exploit_temperature
        self.thinking_budget_fraction = thinking_budget_fraction
        self.chat_token_limit = chat_token_limit
        self.batch_size = batch_size
        
        # Dictionary to store chat sessions: { (island_id, batch_id): chat_object }
        self.chats: dict[tuple[int, int], genai.chats.AsyncChats] = {}
        
        # Track token usage per chat: { (island_id, batch_id): current_prompt_tokens }
        self.chat_token_counts: dict[tuple[int, int], int] = {}
        
        # ===== COST TRACKING =====
        # Cumulative token counts per chat: { (island_id, batch_id): total_tokens_ever }
        self.chat_cumulative_tokens: dict[tuple[int, int], int] = {}
        # Tokens used in current iteration per chat
        self.chat_iteration_tokens: dict[tuple[int, int], int] = {}
        # Total resets per chat
        self.chat_reset_counts: dict[tuple[int, int], int] = {}
        # Global counters
        self.total_prompt_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_resets: int = 0
        
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
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=True),
            system_instruction=explore_instruction
        )
        self.large_exploit_config = types.GenerateContentConfig(
            temperature=self.exploit_temperature,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget, include_thoughts=True),
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
        logging.info(f"Chat Token Limit: {self.chat_token_limit} (0 = unlimited)")
        logging.info(f"Active chats: {list(self.chats.keys()) if self.chats else 'None yet'}")
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

    def _create_chat(self, island_id: int, batch_id: int, mode: str = 'explore', 
                      use_large_model: bool = True) -> genai.chats.AsyncChats:
        """
        Create a new chat session for an (island, batch) pair.
        
        Args:
            island_id: The island identifier.
            batch_id: The batch identifier within the island.
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
        
        logging.info(f"Created new chat session for Island {island_id}, Batch {batch_id} "
                    f"(model={model_name}, mode={mode})")
        return chat

    async def get_or_create_chat(self, island_id: int, batch_id: int, mode: str = 'explore',
                                   use_large_model: bool = True) -> genai.chats.AsyncChats:
        """
        Get the chat session for an (island, batch) pair, creating one if it doesn't exist.
        
        Args:
            island_id: The island identifier.
            batch_id: The batch identifier within the island.
            mode: 'explore' or 'exploit' for new chat creation.
            use_large_model: Model size for new chat creation.
            
        Returns:
            The chat object for this (island, batch) pair.
        """
        chat_key = (island_id, batch_id)
        if chat_key not in self.chats:
            self.chats[chat_key] = self._create_chat(island_id, batch_id, mode, use_large_model)
        return self.chats[chat_key]

    async def ask_island(self, island_id: int, prompt: str, 
                         batch_id: int,
                         mode: str = 'explore',
                         use_large_model: bool = True,
                         png_img: Optional[bytes] = None) -> str:
        """
        Send a message to an (island, batch) chat and get a response.
        
        The message is automatically appended to the chat's history,
        allowing the LLM to see and learn from previous interactions
        within this specific (island, batch) lineage.
        
        Uses override config to dynamically switch between model/mode combinations
        without recreating the chat session.
        
        Args:
            island_id: The island identifier.
            prompt: The prompt text to send.
            mode: 'explore' or 'exploit' - affects system instruction and temperature.
            use_large_model: If True, use large model config; otherwise small model.
            batch_id: The batch identifier within the island (default 0).
            png_img: Optional PNG image bytes to include with the prompt.
            
        Returns:
            The LLM's response text, or empty string on error.
        """
        chat_key = (island_id, batch_id)
        chat = await self.get_or_create_chat(island_id, batch_id, mode, use_large_model)
        
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

            # Track token usage
            prompt_tokens = response.usage_metadata.prompt_token_count
            output_tokens = response.usage_metadata.candidates_token_count
            self.chat_token_counts[chat_key] = prompt_tokens
            
            # Update cumulative tracking
            self.chat_cumulative_tokens[chat_key] = self.chat_cumulative_tokens.get(chat_key, 0) + prompt_tokens + output_tokens
            self.chat_iteration_tokens[chat_key] = self.chat_iteration_tokens.get(chat_key, 0) + prompt_tokens + output_tokens
            self.total_prompt_tokens += prompt_tokens
            self.total_output_tokens += output_tokens
            
            logging.info(f"[Island {island_id}, Batch {batch_id}] Tokens: {response.usage_metadata.total_token_count} (in={prompt_tokens}, out={output_tokens})")
            
            # Check if we've exceeded the token limit
            if self.chat_token_limit > 0 and prompt_tokens > self.chat_token_limit:
                logging.info(f"Chat ({island_id}, {batch_id}) exceeded token limit ({prompt_tokens} > {self.chat_token_limit}). Summarizing and resetting...")
                print(f"Chat ({island_id}, {batch_id}) exceeded token limit ({prompt_tokens} > {self.chat_token_limit}). Summarizing and resetting...")
                await self.summarize_and_reset_chat(island_id, batch_id, mode, use_large_model)

            return response.text
        except Exception as e:
            model_name = self.get_model_name(use_large_model)
            print(f"Error sending message to chat ({island_id}, {batch_id}) "
                  f"(model={model_name}, mode={mode}): {e}")
            return ""

    async def get_chat_history(self, island_id: int, batch_id: int = 0) -> list:
        """
        Retrieve the chat history for an (island, batch) pair.
        
        Args:
            island_id: The island identifier.
            batch_id: The batch identifier within the island.
            
        Returns:
            The chat history list, or empty list if not found/error.
        """
        chat_key = (island_id, batch_id)
        try:
            if chat_key in self.chats:
                chat = self.chats[chat_key]
                # get_history() may or may not be async depending on SDK version
                history = chat.get_history()
                # If it's a coroutine, await it
                if hasattr(history, '__await__'):
                    history = await history
                return list(history) if history else []
            else:
                print(f"No chat found for ({island_id}, {batch_id})")
                return []
        except Exception as e:
            print(f"Error retrieving chat history for ({island_id}, {batch_id}): {e}")
            return []
    
    async def summarize_and_reset_chat(self, island_id: int, batch_id: int = 0,
                                         mode: str = 'explore',
                                         use_large_model: bool = False) -> Optional[str]:
        """
        Compresses the history of an (island, batch) chat into a summary and starts fresh.
        
        Called automatically when token usage exceeds chat_token_limit.
        
        Args:
            island_id: The island identifier.
            batch_id: The batch identifier within the island.
            mode: Current mode for the new chat session.
            use_large_model: Whether to use large model for the new session.
            
        Returns:
            The summary text, or None if chat doesn't exist.
        """
        chat_key = (island_id, batch_id)
        if chat_key not in self.chats:
            return None

        old_chat = self.chats[chat_key]
        old_token_count = self.chat_token_counts.get(chat_key, 0)
        
        # 1. Ask for a technical summary using small model config (cheaper)
        summary_prompt = (
            "Summarize our conversation so far. Provide a concise technical summary of:\n"
            "1. The best-performing model architectures we've discovered.\n"
            "2. Key insights about what features correlate with lower loss.\n"
            "3. Key insights about what improved parameter estimator performance.\n"
            "4. Approaches that did NOT work well (to avoid repeating).\n"
            "5. The current best loss achieved.\n"
            "Be concise - this summary will seed a fresh conversation."
        )
        
        try:
            summary_config = self.get_config('exploit', use_large_model=False)  # Use small model for summary
            response = await old_chat.send_message(summary_prompt, config=summary_config)
            summary_text = response.text
            
            logging.info(f"Chat ({island_id}, {batch_id}) summary generated ({len(summary_text)} chars)")
            logging.info(f"Summary: {summary_text[:500]}..." if len(summary_text) > 500 else f"Summary: {summary_text}")
        except Exception as e:
            logging.info(f"Error generating summary for chat ({island_id}, {batch_id}): {e}")
            summary_text = "Previous session context unavailable."

        # 2. Create a new chat with the summary as context
        # Get the base system instruction and append the summary
        base_instruction = self.get_system_instruction(mode)
        enhanced_instruction = (
            f"{base_instruction}\n\n"
            f"# CONTEXT FROM PREVIOUS SESSION\n"
            f"The following is a summary of our previous work on this lineage:\n"
            f"{summary_text}"
        )
        
        # Create new config with enhanced system instruction
        config = self.get_config(mode, use_large_model)
        # We need to create a new config with the enhanced instruction
        if use_large_model:
            new_config = types.GenerateContentConfig(
                temperature=config.temperature,
                thinking_config=config.thinking_config,
                system_instruction=enhanced_instruction
            )
        else:
            new_config = types.GenerateContentConfig(
                temperature=config.temperature,
                system_instruction=enhanced_instruction
            )

        # 3. Replace the old chat with a brand new one (clears history)
        self.chats[chat_key] = self.client.aio.chats.create(
            model=self.get_model_name(use_large_model),
            config=new_config,
            history=[]
        )
        
        # Reset token count for this chat
        self.chat_token_counts[chat_key] = 0
        
        # Track reset counts
        self.chat_reset_counts[chat_key] = self.chat_reset_counts.get(chat_key, 0) + 1
        self.total_resets += 1
        
        logging.info(f"Chat ({island_id}, {batch_id}) reset: {old_token_count} tokens -> 0 tokens (reset #{self.chat_reset_counts[chat_key]})")
        print(f"Chat ({island_id}, {batch_id}) has been reset with summary checkpoint (reset #{self.chat_reset_counts[chat_key]}).")
        return summary_text

    def get_n_chats(self) -> int:
        """Return the number of active chats."""
        return len(self.chats)
    
    def reset_chat(self, island_id: int, batch_id: int = 0, mode: str = 'explore', 
                   use_large_model: bool = True) -> None:
        """
        Reset a chat's history by creating a fresh chat.
        
        Useful if the chat history grows too large or needs to be cleared.
        
        Args:
            island_id: The island identifier.
            batch_id: The batch identifier within the island.
            mode: 'explore' or 'exploit' for the new chat.
            use_large_model: Model size for the new chat.
        """
        chat_key = (island_id, batch_id)
        if chat_key in self.chats:
            self.chats[chat_key] = self._create_chat(island_id, batch_id, mode, use_large_model)
            logging.info(f"Reset chat session for ({island_id}, {batch_id})")
    
    def reset_all_chats(self, mode: str = 'explore', use_large_model: bool = True) -> None:
        """Reset all chat histories."""
        for chat_key in list(self.chats.keys()):
            island_id, batch_id = chat_key
            self.reset_chat(island_id, batch_id, mode, use_large_model)

    def start_iteration(self) -> None:
        """
        Call at the start of each iteration to reset per-iteration token tracking.
        """
        self.chat_iteration_tokens = {}
    
    def log_iteration_summary(self, iteration: int) -> None:
        """
        Log token usage summary for the current iteration with visual $ borders.
        Call this at the end of each iteration.
        
        Args:
            iteration: The current iteration number.
        """
        width = 70
        border = "$" * width
        
        logging.info(border)
        logging.info(f"${'TOKEN USAGE - ITERATION ' + str(iteration):^{width-2}}$")
        logging.info(border)
        
        # Per-chat stats for this iteration (aggregate by island for readability)
        island_tokens = {}  # island_id -> total tokens this iteration
        total_iter_tokens = 0
        for chat_key in sorted(self.chat_iteration_tokens.keys()):
            tokens = self.chat_iteration_tokens[chat_key]
            total_iter_tokens += tokens
            island_id = chat_key[0]
            island_tokens[island_id] = island_tokens.get(island_id, 0) + tokens
        
        for island_id in sorted(island_tokens.keys()):
            tokens = island_tokens[island_id]
            # Count resets for all batches of this island
            resets = sum(self.chat_reset_counts.get((island_id, b), 0) for b in range(self.batch_size))
            current = sum(self.chat_token_counts.get((island_id, b), 0) for b in range(self.batch_size))
            logging.info(f"$ Island {island_id}: {tokens:,} tokens this iter, {current:,} in context, {resets} resets $")
        
        logging.info(f"${'':^{width-2}}$")
        logging.info(f"$ {'Iteration Total:':<20} {total_iter_tokens:>15,} tokens{' ':>17}$")
        logging.info(f"$ {'Cumulative Total:':<20} {self.total_prompt_tokens + self.total_output_tokens:>15,} tokens{' ':>17}$")
        logging.info(f"$ {'Total Resets:':<20} {self.total_resets:>15}{' ':>23}$")
        logging.info(border)
        
        # Also print to console
        print(border)
        print(f"TOKEN USAGE - ITERATION {iteration}")
        print(f"  Iteration tokens: {total_iter_tokens:,}")
        print(f"  Cumulative tokens: {self.total_prompt_tokens + self.total_output_tokens:,}")
        print(f"  Total resets: {self.total_resets}")
        print(border)
    
    def log_final_summary(self) -> None:
        """
        Log final token usage summary at the end of the run with visual $ borders.
        """
        width = 70
        border = "$" * width
        
        logging.info("")
        logging.info(border)
        logging.info(f"${'FINAL TOKEN USAGE SUMMARY':^{width-2}}$")
        logging.info(border)
        
        total_tokens = self.total_prompt_tokens + self.total_output_tokens
        
        # Global stats
        logging.info(f"$ {'Total Prompt Tokens:':<25} {self.total_prompt_tokens:>20,}{' ':>21}$")
        logging.info(f"$ {'Total Output Tokens:':<25} {self.total_output_tokens:>20,}{' ':>21}$")
        logging.info(f"$ {'TOTAL TOKENS:':<25} {total_tokens:>20,}{' ':>21}$")
        logging.info(f"${'':^{width-2}}$")
        logging.info(f"$ {'Total Resets Triggered:':<25} {self.total_resets:>20}{' ':>21}$")
        logging.info(f"${'':^{width-2}}$")
        
        # Per-island breakdown (aggregate across batches)
        logging.info(f"${'PER-ISLAND BREAKDOWN':^{width-2}}$")
        logging.info(f"${'':^{width-2}}$")
        
        # Get unique island IDs
        island_ids = set(chat_key[0] for chat_key in self.chat_cumulative_tokens.keys())
        for island_id in sorted(island_ids):
            cumulative = sum(self.chat_cumulative_tokens.get((island_id, b), 0) for b in range(self.batch_size))
            resets = sum(self.chat_reset_counts.get((island_id, b), 0) for b in range(self.batch_size))
            logging.info(f"$   Island {island_id}: {cumulative:>15,} tokens, {resets:>3} resets{' ':>30}$")
        
        logging.info(border)
        
        # Also print to console
        print("")
        print(border)
        print(f"{'FINAL TOKEN USAGE SUMMARY':^{width}}")
        print(border)
        print(f"  Total Prompt Tokens:  {self.total_prompt_tokens:>15,}")
        print(f"  Total Output Tokens:  {self.total_output_tokens:>15,}")
        print(f"  TOTAL TOKENS:         {total_tokens:>15,}")
        print(f"  Total Resets:         {self.total_resets:>15}")
        print("")
        print("  Per-Island Breakdown:")
        for island_id in sorted(island_ids):
            cumulative = sum(self.chat_cumulative_tokens.get((island_id, b), 0) for b in range(self.batch_size))
            resets = sum(self.chat_reset_counts.get((island_id, b), 0) for b in range(self.batch_size))
            print(f"    Island {island_id}: {cumulative:>12,} tokens, {resets} resets")
        print(border)

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
