"""
Test script to verify that overriding configs in ask_island() preserves chat history.

This tests the concern that switching between configs (different modes/models)
mid-conversation might reset or lose the chat history.

Run with: python -m tests.test_chat_history_persistence
"""

import asyncio
import os
from dotenv import load_dotenv
from google import genai

# Add parent directory to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.llm_helper import IslandChatManager


def mock_get_system_instruction(mode: str) -> str:
    """Simple mock system instruction generator for testing."""
    if mode == 'explore':
        return "You are a helpful assistant in EXPLORE mode. Be creative and inventive."
    else:
        return "You are a helpful assistant in EXPLOIT mode. Be focused and precise."


async def test_chat_history_persistence():
    """
    Test that switching configs mid-conversation preserves chat history.
    
    Steps:
    1. Create IslandChatManager
    2. Send message in explore mode with large model
    3. Check history length
    4. Send message in exploit mode with small model (config switch!)
    5. Verify history still contains all messages
    6. Send another message back in explore mode
    7. Verify complete history is preserved
    """
    print("=" * 60)
    print("TEST: Chat History Persistence Across Config Switches")
    print("=" * 60)
    
    # Load API key
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return False
    
    client = genai.Client(api_key=api_key)
    
    # Create IslandChatManager with test configs
    manager = IslandChatManager(
        client=client,
        get_system_instruction=mock_get_system_instruction,
        small_model_name="gemini-2.0-flash",
        large_model_name="gemini-2.0-flash",  # Use same model to avoid model differences
        explore_temperature=1.0,
        exploit_temperature=0.5,
        thinking_budget_fraction=0.0  # No thinking for faster tests
    )
    
    island_id = 0
    
    print("\n--- Step 1: Send first message (explore mode, large model) ---")
    response1 = await manager.ask_island(
        island_id=island_id,
        prompt="Hello! My name is TestUser. Please remember my name.",
        mode='explore',
        use_large_model=True
    )
    print(f"Response 1: {response1[:200]}..." if len(response1) > 200 else f"Response 1: {response1}")
    
    # Check history after first message
    history1 = await manager.get_island_history(island_id)
    print(f"\nHistory length after message 1: {len(history1)} entries")
    
    print("\n--- Step 2: Send second message (SWITCH to exploit mode, small model) ---")
    response2 = await manager.ask_island(
        island_id=island_id,
        prompt="What is my name? You should remember it from my previous message.",
        mode='exploit',  # Different mode!
        use_large_model=False  # Different model size!
    )
    print(f"Response 2: {response2[:200]}..." if len(response2) > 200 else f"Response 2: {response2}")
    
    # Check if the model remembered the name (indicates history was preserved)
    name_remembered = "testuser" in response2.lower()
    print(f"\nName 'TestUser' found in response: {name_remembered}")
    
    # Check history after second message
    history2 = await manager.get_island_history(island_id)
    print(f"History length after message 2: {len(history2)} entries")
    
    print("\n--- Step 3: Send third message (SWITCH back to explore mode, large model) ---")
    response3 = await manager.ask_island(
        island_id=island_id,
        prompt="Can you tell me what we've discussed so far? Summarize our conversation.",
        mode='explore',  # Back to explore
        use_large_model=True  # Back to large
    )
    print(f"Response 3: {response3[:300]}..." if len(response3) > 300 else f"Response 3: {response3}")
    
    # Check history after third message
    history3 = await manager.get_island_history(island_id)
    print(f"\nHistory length after message 3: {len(history3)} entries")
    
    # Verify history content
    print("\n--- History Verification ---")
    print(f"Expected history entries: 6 (3 user messages + 3 assistant responses)")
    print(f"Actual history entries: {len(history3)}")
    
    # Print history roles
    if history3:
        print("\nHistory roles in order:")
        for i, entry in enumerate(history3):
            role = getattr(entry, 'role', 'unknown')
            # Get a snippet of the text
            parts = getattr(entry, 'parts', [])
            text_snippet = ""
            if parts:
                text = getattr(parts[0], 'text', '')
                text_snippet = text[:50] + "..." if len(text) > 50 else text
            print(f"  {i+1}. {role}: {text_snippet}")
    
    # Final verdict
    print("\n" + "=" * 60)
    print("TEST RESULTS:")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 3
    
    # Test 1: History grows with each message
    if len(history3) >= 6:
        print("✅ PASS: History length is correct (6+ entries)")
        tests_passed += 1
    else:
        print(f"❌ FAIL: History length is {len(history3)}, expected 6+")
    
    # Test 2: Name was remembered after config switch
    if name_remembered:
        print("✅ PASS: Model remembered context after config switch")
        tests_passed += 1
    else:
        print("❌ FAIL: Model did NOT remember context after config switch")
    
    # Test 3: History length increased after each message
    if len(history1) < len(history2) < len(history3):
        print("✅ PASS: History grew after each message")
        tests_passed += 1
    else:
        print(f"❌ FAIL: History did not grow properly: {len(history1)} -> {len(history2)} -> {len(history3)}")
    
    print(f"\nOverall: {tests_passed}/{tests_total} tests passed")
    print("=" * 60)
    
    return tests_passed == tests_total


async def test_multiple_islands_independent():
    """
    Test that different islands maintain independent histories.
    """
    print("\n" + "=" * 60)
    print("TEST: Multiple Islands Have Independent Histories")
    print("=" * 60)
    
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment")
        return False
    
    client = genai.Client(api_key=api_key)
    
    manager = IslandChatManager(
        client=client,
        get_system_instruction=mock_get_system_instruction,
        small_model_name="gemini-2.0-flash",
        large_model_name="gemini-2.0-flash",
        explore_temperature=1.0,
        exploit_temperature=0.5,
        thinking_budget_fraction=0.0
    )
    
    # Send different messages to different islands
    print("\n--- Sending to Island 0 ---")
    await manager.ask_island(0, "I am on Island ZERO.", mode='explore', use_large_model=True)
    
    print("--- Sending to Island 1 ---")
    await manager.ask_island(1, "I am on Island ONE.", mode='explore', use_large_model=True)
    
    # Check histories are independent
    history0 = await manager.get_island_history(0)
    history1 = await manager.get_island_history(1)
    
    print(f"\nIsland 0 history length: {len(history0)}")
    print(f"Island 1 history length: {len(history1)}")
    
    # Each should have exactly 2 entries (1 user + 1 assistant)
    if len(history0) == 2 and len(history1) == 2:
        print("✅ PASS: Islands have independent histories")
        return True
    else:
        print("❌ FAIL: Islands do not have independent histories")
        return False


if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("# Chat History Persistence Test Suite")
    print("#" * 60)
    
    # Run tests
    result1 = asyncio.run(test_chat_history_persistence())
    result2 = asyncio.run(test_multiple_islands_independent())
    
    print("\n" + "#" * 60)
    print("# FINAL SUMMARY")
    print("#" * 60)
    if result1 and result2:
        print("✅ ALL TESTS PASSED - Chat history is preserved across config switches!")
    else:
        print("❌ SOME TESTS FAILED - There may be issues with history persistence")
    print("#" * 60)
