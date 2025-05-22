from app.core.config import settings
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage # Correct import path
from typing import List

# Note: The global redis_client and its initialization are removed as
# RedisChatMessageHistory handles its own Redis connections.

def _get_langchain_history_store(session_id: str) -> RedisChatMessageHistory:
    """
    Helper function to instantiate RedisChatMessageHistory.
    The TTL for the session is handled by RedisChatMessageHistory itself,
    using the underlying Redis EXPIRE command.
    """
    return RedisChatMessageHistory(
        session_id=session_id,
        url=settings.REDIS_URL, # Uses the new REDIS_URL from config
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )

def add_message_to_history(session_id: str, user_message: str, ai_message: str) -> bool:
    """
    Stores a pair of user message and AI response using RedisChatMessageHistory.
    """
    try:
        history = _get_langchain_history_store(session_id)
        history.add_messages([
            HumanMessage(content=user_message),
            AIMessage(content=ai_message)
        ])
        # TTL is managed by RedisChatMessageHistory on add/update.
        print(f"Messages added to Langchain history for session {session_id}. TTL set/refreshed.")
        return True
    except Exception as e: # Catching general exception as Redis client errors might vary
        print(f"Error adding messages to Langchain history for session {session_id}: {e}")
        return False

def get_conversation_history(session_id: str) -> List[BaseMessage]:
    """
    Retrieves the conversation history for the given session_id using RedisChatMessageHistory.
    Returns a list of BaseMessage objects (HumanMessage, AIMessage).
    """
    try:
        history = _get_langchain_history_store(session_id)
        messages = history.messages
        if messages:
            print(f"Retrieved {len(messages)} messages from Langchain history for session {session_id}.")
        else:
            print(f"No Langchain conversation history found for session {session_id} (or key expired).")
        return messages
    except Exception as e:
        print(f"Error retrieving Langchain history for session {session_id}: {e}")
        return []

def clear_conversation_history(session_id: str) -> bool:
    """
    Deletes the conversation history for the given session_id using RedisChatMessageHistory.
    """
    try:
        history = _get_langchain_history_store(session_id)
        history.clear()
        print(f"Successfully cleared Langchain conversation history for session {session_id}.")
        return True
    except Exception as e:
        print(f"Error clearing Langchain history for session {session_id}: {e}")
        return False

# --- Example Usage (Updated for LangChain) ---
if __name__ == "__main__":
    import time
    print("\n--- Running conversation_manager.py example (LangChain version) ---")

    # Ensure Redis is running and accessible via settings.REDIS_URL
    # Test basic connectivity conceptually (RedisChatMessageHistory handles actual connection)
    try:
        # A quick check to see if Redis is up, not directly using the history object yet for this
        import redis
        r = redis.from_url(settings.REDIS_URL)
        r.ping()
        print(f"Successfully pinged Redis at {settings.REDIS_URL}")
    except Exception as e:
        print(f"Could not connect to Redis at {settings.REDIS_URL}. Ensure Redis is running. Error: {e}")
        print("Exiting example as Redis connection is required.")
        exit()

    test_session_id_lc = "test_session_lc_123"
    original_timeout_lc = settings.CONVERSATION_TIMEOUT_SECONDS

    # 0. Clean up any pre-existing test session data
    print(f"\n--- Cleaning up pre-existing session: {test_session_id_lc} ---")
    clear_conversation_history(test_session_id_lc)

    # 1. Add messages to history
    print("\n--- Adding Messages to History (LangChain) ---")
    add_message_to_history(test_session_id_lc, "Hello AI, from LangChain!", "Hello User! This is LangChain's AIMessage.")
    time.sleep(0.1)
    add_message_to_history(test_session_id_lc, "How does LangChain handle Redis history?", "LangChain uses RedisChatMessageHistory for that!")

    # 2. Retrieve conversation history
    print("\n--- Retrieving Conversation History (LangChain) ---")
    lc_history_messages = get_conversation_history(test_session_id_lc)
    if lc_history_messages:
        print(f"Retrieved history for {test_session_id_lc}:")
        for i, msg in enumerate(lc_history_messages):
            print(f"  Message {i+1}: Type: {type(msg).__name__}, Content: '{msg.content}'")
    else:
        print(f"No history found for {test_session_id_lc} after adding messages.")

    # 3. Demonstrate Expiration (shortened for example, using LangChain's TTL)
    print("\n--- Demonstrating Expiration with LangChain TTL (Conceptual) ---")
    settings.CONVERSATION_TIMEOUT_SECONDS = 3 # 3 seconds for TTL
    print(f"Temporarily set CONVERSATION_TIMEOUT_SECONDS (TTL) to: {settings.CONVERSATION_TIMEOUT_SECONDS}s")
    
    # Add another message to refresh with the new short TTL
    add_message_to_history(test_session_id_lc, "Testing LangChain TTL.", "This message should make the session expire soon.")
    
    print(f"Waiting for {settings.CONVERSATION_TIMEOUT_SECONDS + 2} seconds to check for expiration...") # Wait a bit longer than TTL
    time.sleep(settings.CONVERSATION_TIMEOUT_SECONDS + 2)
    
    history_after_expiry_attempt_lc = get_conversation_history(test_session_id_lc)
    if not history_after_expiry_attempt_lc:
        print(f"LangChain history for {test_session_id_lc} has expired as expected.")
    else:
        print(f"LangChain history for {test_session_id_lc} still exists ({len(history_after_expiry_attempt_lc)} messages). Expiration might not have triggered as expected or Redis TTL granularity.")
        for msg in history_after_expiry_attempt_lc:
            print(f"    Existing Msg: Type: {type(msg).__name__}, Content: '{msg.content}'")


    # Restore original timeout for subsequent tests or operations
    settings.CONVERSATION_TIMEOUT_SECONDS = original_timeout_lc
    print(f"Restored CONVERSATION_TIMEOUT_SECONDS to: {settings.CONVERSATION_TIMEOUT_SECONDS}s")

    # Re-populate for clearing test if it already expired
    if not history_after_expiry_attempt_lc:
        print("\n--- Re-populating history for clearing test (LangChain) ---")
        add_message_to_history(test_session_id_lc, "Hello again, LangChain!", "Hi there, from AIMessage!")

    # 4. Clear conversation history
    print("\n--- Clearing Conversation History (LangChain) ---")
    if clear_conversation_history(test_session_id_lc):
        print(f"LangChain history for {test_session_id_lc} cleared successfully.")
    else:
        print(f"Failed to clear LangChain history for {test_session_id_lc} (it might have already expired or never existed).")

    # 5. Verify history is cleared
    print("\n--- Verifying History is Cleared (LangChain) ---")
    history_after_clear_lc = get_conversation_history(test_session_id_lc)
    if not history_after_clear_lc:
        print(f"LangChain history for {test_session_id_lc} is confirmed cleared.")
    else:
        print(f"LangChain history for {test_session_id_lc} was not cleared successfully. Found {len(history_after_clear_lc)} messages.")

    print("\n--- conversation_manager.py example (LangChain version) finished ---")
