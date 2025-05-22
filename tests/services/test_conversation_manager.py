import pytest
from unittest.mock import MagicMock, patch # For more complex mocking if needed

# Modules and classes to test or use in tests
from app.services import conversation_manager as cm
from app.core.config import settings
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
# RedisChatMessageHistory is what we'll be mocking
from langchain_community.chat_message_histories import RedisChatMessageHistory 

# Note: No global state to reset in conversation_manager.py itself, as RedisChatMessageHistory
# instances are created per function call. Each test will mock this class.

@pytest.fixture
def mock_redis_chat_message_history(mocker):
    """
    Fixture to mock the RedisChatMessageHistory class.
    It will return a MagicMock instance when RedisChatMessageHistory is called.
    The instance itself will also be a MagicMock, allowing inspection of method calls on it.
    """
    # Create a mock for the RedisChatMessageHistory class itself
    mock_class = mocker.MagicMock(spec=RedisChatMessageHistory)
    
    # Create a mock for instances that RedisChatMessageHistory() would return
    mock_instance = mocker.MagicMock(spec=RedisChatMessageHistory)
    mock_instance.messages = [] # Default for .messages property
    
    # Configure the mock class to return the mock instance upon instantiation
    mock_class.return_value = mock_instance
    
    # Patch the class in the conversation_manager module where it's imported
    mocker.patch('app.services.conversation_manager.RedisChatMessageHistory', mock_class)
    
    return mock_class, mock_instance


def test_add_message_to_history_success(mock_redis_chat_message_history):
    """Test successfully adding a message to history."""
    mock_class, mock_instance = mock_redis_chat_message_history

    session_id = "test_session_add"
    user_msg = "Hello there!"
    ai_msg = "General Kenobi!"

    result = cm.add_message_to_history(session_id, user_msg, ai_msg)

    assert result is True
    # Check RedisChatMessageHistory instantiation
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )
    # Check that add_messages was called on the instance
    mock_instance.add_messages.assert_called_once()
    args, _ = mock_instance.add_messages.call_args
    messages_arg = args[0]
    assert len(messages_arg) == 2
    assert isinstance(messages_arg[0], HumanMessage)
    assert messages_arg[0].content == user_msg
    assert isinstance(messages_arg[1], AIMessage)
    assert messages_arg[1].content == ai_msg


def test_add_message_to_history_failure(mock_redis_chat_message_history):
    """Test failure when adding a message to history (e.g., Redis error)."""
    mock_class, mock_instance = mock_redis_chat_message_history
    mock_instance.add_messages.side_effect = Exception("Redis unavailable")

    session_id = "test_session_add_fail"
    result = cm.add_message_to_history(session_id, "User", "AI")

    assert result is False
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )
    mock_instance.add_messages.assert_called_once() # Method was called


def test_get_conversation_history_success(mock_redis_chat_message_history):
    """Test successfully retrieving conversation history."""
    mock_class, mock_instance = mock_redis_chat_message_history
    
    session_id = "test_session_get"
    expected_messages = [
        HumanMessage(content="Question 1"),
        AIMessage(content="Answer 1")
    ]
    # Configure the .messages property of the mock instance
    # For properties, assign directly to the mock_instance if it's a simple list,
    # or use mocker.PropertyMock if more complex behavior is needed.
    # Here, RedisChatMessageHistory.messages is a property that returns a list.
    type(mock_instance).messages = pytest.PropertyMock(return_value=expected_messages)


    retrieved_messages = cm.get_conversation_history(session_id)

    assert retrieved_messages == expected_messages
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS # TTL is set even on retrieval for consistency if class does it
    )
    # Verify that the 'messages' property was accessed
    assert type(mock_instance).messages.called


def test_get_conversation_history_empty(mock_redis_chat_message_history):
    """Test retrieving an empty conversation history."""
    mock_class, mock_instance = mock_redis_chat_message_history
    type(mock_instance).messages = pytest.PropertyMock(return_value=[]) # Empty history

    session_id = "test_session_get_empty"
    retrieved_messages = cm.get_conversation_history(session_id)

    assert retrieved_messages == []
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )
    assert type(mock_instance).messages.called


def test_get_conversation_history_failure(mock_redis_chat_message_history):
    """Test failure when retrieving conversation history."""
    mock_class, mock_instance = mock_redis_chat_message_history
    # Simulate error when accessing the .messages property
    type(mock_instance).messages = pytest.PropertyMock(side_effect=Exception("Redis read error"))

    session_id = "test_session_get_fail"
    retrieved_messages = cm.get_conversation_history(session_id)

    assert retrieved_messages == [] # Should return empty list on failure
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )
    assert type(mock_instance).messages.called


def test_clear_conversation_history_success(mock_redis_chat_message_history):
    """Test successfully clearing conversation history."""
    mock_class, mock_instance = mock_redis_chat_message_history

    session_id = "test_session_clear"
    result = cm.clear_conversation_history(session_id)

    assert result is True
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )
    mock_instance.clear.assert_called_once()


def test_clear_conversation_history_failure(mock_redis_chat_message_history):
    """Test failure when clearing conversation history."""
    mock_class, mock_instance = mock_redis_chat_message_history
    mock_instance.clear.side_effect = Exception("Redis clear error")

    session_id = "test_session_clear_fail"
    result = cm.clear_conversation_history(session_id)

    assert result is False
    mock_class.assert_called_once_with(
        session_id=session_id,
        url=settings.REDIS_URL,
        ttl=settings.CONVERSATION_TIMEOUT_SECONDS
    )
    mock_instance.clear.assert_called_once()

# To run: pytest tests/services/test_conversation_manager.py
# Ensure PYTHONPATH includes the project root or use appropriate pytest configuration.
# Example: PYTHONPATH=. pytest tests/services/test_conversation_manager.py
