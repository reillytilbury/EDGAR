"""
Tests for IslandChatManager token limit and reset functionality.

Tests that:
1. Token counts are tracked per island
2. summarize_and_reset_island resets token count to 0
3. System instruction is updated with summary context after reset
"""
import asyncio
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from google.genai import types


# =============================================================================
# MOCK SETUP
# =============================================================================

def create_mock_response(prompt_tokens: int = 1000, output_tokens: int = 100, text: str = "response"):
    """Create a mock response with usage metadata."""
    response = Mock()
    response.text = text
    response.usage_metadata = Mock()
    response.usage_metadata.prompt_token_count = prompt_tokens
    response.usage_metadata.candidates_token_count = output_tokens
    response.usage_metadata.total_token_count = prompt_tokens + output_tokens
    return response


def create_mock_chat(responses: list = None):
    """Create a mock chat object that returns the given responses in sequence."""
    if responses is None:
        responses = [create_mock_response()]
    
    chat = AsyncMock()
    chat.send_message = AsyncMock(side_effect=responses)
    chat.get_history = Mock(return_value=[])
    return chat


def create_mock_client(mock_chat):
    """Create a mock genai client."""
    client = Mock()
    client.aio = Mock()
    client.aio.chats = Mock()
    client.aio.chats.create = Mock(return_value=mock_chat)
    return client


def mock_get_system_instruction(mode: str) -> str:
    """Mock system instruction generator."""
    if mode == 'explore':
        return "You are a creative scientist exploring new hypotheses."
    else:
        return "You are a focused scientist refining existing hypotheses."


# =============================================================================
# TESTS
# =============================================================================

class TestTokenTracking:
    """Tests for token usage tracking."""
    
    @pytest.mark.asyncio
    async def test_token_count_tracked_after_response(self):
        """Token count should be recorded after each response."""
        from src.llm_helper import IslandChatManager
        
        expected_tokens = 5000
        mock_chat = create_mock_chat([create_mock_response(prompt_tokens=expected_tokens)])
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        await manager.ask_island(island_id=0, prompt="test prompt")
        
        assert manager.island_token_counts[0] == expected_tokens
    
    @pytest.mark.asyncio
    async def test_token_count_updates_on_each_call(self):
        """Token count should update with each response's prompt_token_count."""
        from src.llm_helper import IslandChatManager
        
        # Simulate growing token count (as history accumulates)
        responses = [
            create_mock_response(prompt_tokens=1000),
            create_mock_response(prompt_tokens=3000),
            create_mock_response(prompt_tokens=6000),
        ]
        mock_chat = create_mock_chat(responses)
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        await manager.ask_island(island_id=0, prompt="first")
        assert manager.island_token_counts[0] == 1000
        
        await manager.ask_island(island_id=0, prompt="second")
        assert manager.island_token_counts[0] == 3000
        
        await manager.ask_island(island_id=0, prompt="third")
        assert manager.island_token_counts[0] == 6000


class TestSummarizeAndReset:
    """Tests for the summarize_and_reset_island functionality."""
    
    @pytest.mark.asyncio
    async def test_reset_reduces_token_count_to_zero(self):
        """After reset, token count for the island should be 0."""
        from src.llm_helper import IslandChatManager
        
        # First response builds up tokens, second is the summary response
        responses = [
            create_mock_response(prompt_tokens=40000),
            create_mock_response(prompt_tokens=42000, text="Summary: best models were X, Y, Z...")
        ]
        mock_chat = create_mock_chat(responses)
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        # Build up token count
        await manager.ask_island(island_id=0, prompt="test")
        assert manager.island_token_counts[0] == 40000
        
        # Manually trigger reset
        await manager.summarize_and_reset_island(island_id=0, mode='explore', use_large_model=False)
        
        # Token count should be reset to 0
        assert manager.island_token_counts[0] == 0
    
    @pytest.mark.asyncio
    async def test_reset_creates_new_chat_with_enhanced_instruction(self):
        """After reset, the new chat should have summary in system instruction."""
        from src.llm_helper import IslandChatManager
        
        summary_text = "Best model: quadratic. Avoid: exponential functions."
        responses = [
            create_mock_response(prompt_tokens=40000),
            create_mock_response(text=summary_text)  # Summary response
        ]
        mock_chat = create_mock_chat(responses)
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        # Create island and build token count
        await manager.ask_island(island_id=0, prompt="test")
        
        # Reset the island
        returned_summary = await manager.summarize_and_reset_island(
            island_id=0, mode='explore', use_large_model=False
        )
        
        # Verify the summary was returned
        assert returned_summary == summary_text
        
        # Verify client.aio.chats.create was called to create the new chat
        # The second call should have an enhanced system instruction
        calls = mock_client.aio.chats.create.call_args_list
        
        # There should be at least 2 calls: initial creation + reset
        assert len(calls) >= 2
        
        # Check the last call (the reset) has the enhanced instruction
        last_call = calls[-1]
        config = last_call.kwargs.get('config') or last_call[1].get('config')
        
        assert config is not None
        assert "CONTEXT FROM PREVIOUS SESSION" in config.system_instruction
        assert summary_text in config.system_instruction
    
    @pytest.mark.asyncio
    async def test_reset_returns_none_for_nonexistent_island(self):
        """Resetting a non-existent island should return None."""
        from src.llm_helper import IslandChatManager
        
        mock_chat = create_mock_chat()
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
        )
        
        result = await manager.summarize_and_reset_island(island_id=999)
        assert result is None


class TestAutoResetOnTokenLimit:
    """Tests for automatic reset when token limit is exceeded."""
    
    @pytest.mark.asyncio
    async def test_auto_reset_triggered_when_limit_exceeded(self):
        """When prompt_tokens > chat_token_limit, summarize_and_reset should be called."""
        from src.llm_helper import IslandChatManager
        
        # Response that exceeds the limit, then summary response
        responses = [
            create_mock_response(prompt_tokens=60000),  # Exceeds 50k limit
            create_mock_response(text="Summary...")  # Summary response
        ]
        mock_chat = create_mock_chat(responses)
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        # Spy on summarize_and_reset_island
        with patch.object(manager, 'summarize_and_reset_island', wraps=manager.summarize_and_reset_island) as mock_reset:
            await manager.ask_island(island_id=0, prompt="big prompt")
            
            # Verify reset was called
            mock_reset.assert_called_once_with(0, 'explore', True)
    
    @pytest.mark.asyncio
    async def test_no_reset_when_under_limit(self):
        """When prompt_tokens <= chat_token_limit, no reset should occur."""
        from src.llm_helper import IslandChatManager
        
        responses = [
            create_mock_response(prompt_tokens=30000),  # Under 50k limit
        ]
        mock_chat = create_mock_chat(responses)
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        with patch.object(manager, 'summarize_and_reset_island') as mock_reset:
            await manager.ask_island(island_id=0, prompt="normal prompt")
            
            # Verify reset was NOT called
            mock_reset.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_no_reset_when_limit_is_zero(self):
        """When chat_token_limit=0 (unlimited), no reset should occur."""
        from src.llm_helper import IslandChatManager
        
        responses = [
            create_mock_response(prompt_tokens=100000),  # Very high tokens
        ]
        mock_chat = create_mock_chat(responses)
        mock_client = create_mock_client(mock_chat)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=0  # Unlimited
        )
        
        with patch.object(manager, 'summarize_and_reset_island') as mock_reset:
            await manager.ask_island(island_id=0, prompt="huge prompt")
            
            # Verify reset was NOT called (unlimited mode)
            mock_reset.assert_not_called()


class TestMultipleIslands:
    """Tests for tracking across multiple islands."""
    
    @pytest.mark.asyncio
    async def test_separate_token_counts_per_island(self):
        """Each island should have its own token count."""
        from src.llm_helper import IslandChatManager
        
        # Create multiple mock chats, one per island
        mock_chats = {}
        def create_chat_for_island(*args, **kwargs):
            island_count = len(mock_chats)
            # Different token counts per island
            tokens = (island_count + 1) * 1000  # 1k, 2k, 3k...
            chat = create_mock_chat([create_mock_response(prompt_tokens=tokens)])
            mock_chats[island_count] = chat
            return chat
        
        mock_client = Mock()
        mock_client.aio = Mock()
        mock_client.aio.chats = Mock()
        mock_client.aio.chats.create = Mock(side_effect=create_chat_for_island)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        await manager.ask_island(island_id=0, prompt="test")
        await manager.ask_island(island_id=1, prompt="test")
        await manager.ask_island(island_id=2, prompt="test")
        
        assert manager.island_token_counts[0] == 1000
        assert manager.island_token_counts[1] == 2000
        assert manager.island_token_counts[2] == 3000
    
    @pytest.mark.asyncio
    async def test_reset_only_affects_target_island(self):
        """Resetting one island should not affect others."""
        from src.llm_helper import IslandChatManager
        
        # Create chats that will be reused
        call_count = [0]
        def create_chat_for_island(*args, **kwargs):
            call_count[0] += 1
            tokens = 10000  # All start with 10k tokens
            # The summary call response
            chat = create_mock_chat([
                create_mock_response(prompt_tokens=tokens),
                create_mock_response(text="Summary...")
            ])
            return chat
        
        mock_client = Mock()
        mock_client.aio = Mock()
        mock_client.aio.chats = Mock()
        mock_client.aio.chats.create = Mock(side_effect=create_chat_for_island)
        
        manager = IslandChatManager(
            client=mock_client,
            get_system_instruction=mock_get_system_instruction,
            chat_token_limit=50000
        )
        
        # Create 3 islands
        await manager.ask_island(island_id=0, prompt="test")
        await manager.ask_island(island_id=1, prompt="test")
        await manager.ask_island(island_id=2, prompt="test")
        
        # All should have 10k tokens
        assert manager.island_token_counts[0] == 10000
        assert manager.island_token_counts[1] == 10000
        assert manager.island_token_counts[2] == 10000
        
        # Reset only island 1
        await manager.summarize_and_reset_island(island_id=1, mode='explore')
        
        # Only island 1 should be reset
        assert manager.island_token_counts[0] == 10000
        assert manager.island_token_counts[1] == 0
        assert manager.island_token_counts[2] == 10000


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
