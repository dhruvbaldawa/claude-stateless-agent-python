# Agile Task Breakdown: Stateless Claude Agent SDK

This document provides a detailed, iteration-by-iteration task breakdown for implementing the Stateless Claude Agent SDK as described in the architectural design document.

---

### 🔄 **Iteration 1: Core Abstractions & Project Setup**

This iteration establishes the foundational project structure and defines the core protocols and data models that will enable stateless agent execution.

---

#### Task 1: Initialize Python Project Structure

Status: **Pending**

**Goal**: Set up a professional Python package structure with all necessary configuration files, development dependencies, and tooling to support modern async development and testing.

**Working Result**: A working Python package that can be installed in development mode with `pip install -e .`, includes all dependencies (anthropic, aiofiles, redis, asyncpg), and has pytest configured for async testing.

**Validation**:
- [ ] `pyproject.toml` exists with package metadata and dependencies
- [ ] `src/claude_agent_sdk/__init__.py` exists and package is importable
- [ ] `pip install -e .` completes without errors
- [ ] `pytest --collect-only` runs without errors
- [ ] `.gitignore` includes Python-specific ignores (\_\_pycache\_\_, \*.pyc, .venv, etc.)

<prompt>
You are implementing the initial project setup for the Stateless Claude Agent SDK Python package. Follow these steps:

1. **Create the package directory structure**:
   ```
   src/claude_agent_sdk/
   ├── __init__.py
   ├── protocols.py (will hold SessionStore protocol)
   ├── models.py (will hold dataclasses)
   ├── stores/ (storage implementations)
   │   ├── __init__.py
   │   ├── file_store.py
   │   ├── redis_store.py
   │   └── postgres_store.py
   ├── executor.py (agent execution logic)
   └── serialization.py (message serialization utilities)

   tests/
   ├── __init__.py
   ├── test_protocols.py
   ├── test_models.py
   ├── test_serialization.py
   └── stores/
       ├── __init__.py
       ├── test_file_store.py
       ├── test_redis_store.py
       └── test_postgres_store.py
   ```

2. **Create `pyproject.toml`** with the following configuration:
   - Package name: `claude-agent-sdk`
   - Version: `0.1.0`
   - Python requirement: `>=3.10`
   - Core dependencies: `anthropic>=0.34.0`, `aiofiles>=23.0.0`
   - Optional dependencies groups:
     - `redis`: `redis[hiredis]>=5.0.0`
     - `postgres`: `asyncpg>=0.29.0`
     - `dev`: `pytest>=7.4.0`, `pytest-asyncio>=0.21.0`, `pytest-cov>=4.1.0`, `mypy>=1.5.0`, `ruff>=0.1.0`
   - Build system: use `hatchling` or `setuptools>=61.0`
   - Configure pytest: `asyncio_mode = "auto"` in tool.pytest.ini_options

3. **Create all `__init__.py` files** in the directory structure (can be empty for now)

4. **Create a basic `src/claude_agent_sdk/__init__.py`** with:
   ```python
   """Claude Agent SDK - Stateless agent execution with pluggable storage."""

   __version__ = "0.1.0"

   # Public API will be exposed here
   __all__ = []
   ```

5. **Update `.gitignore`** to include Python-specific patterns:
   - `__pycache__/`, `*.py[cod]`, `*$py.class`
   - `.pytest_cache/`, `.coverage`, `htmlcov/`
   - `*.egg-info/`, `dist/`, `build/`
   - `.venv/`, `venv/`, `ENV/`
   - `.mypy_cache/`, `.ruff_cache/`

6. **Create a basic `README.md`** with:
   - Project title and description
   - Installation instructions: `pip install -e ".[dev]"`
   - Basic usage example (can be aspirational)
   - Link to architectural design document

7. **Verify the setup**:
   - Run `pip install -e ".[dev]"` from the repository root
   - Run `python -c "import claude_agent_sdk; print(claude_agent_sdk.__version__)"`
   - Run `pytest --collect-only` to verify test discovery works

8. **Create an initial test file** `tests/test_package.py`:
   ```python
   import claude_agent_sdk

   def test_version():
       assert claude_agent_sdk.__version__ == "0.1.0"
   ```

Run the test to ensure everything is working: `pytest tests/test_package.py -v`
</prompt>

---

#### Task 2: Define Core Data Models

Status: **Pending**

**Goal**: Create the foundational dataclasses (`SessionState`, `SessionMetadata`, `CompactionState`) that represent the complete state of an agent session, with proper type annotations and serialization support.

**Working Result**: A `models.py` module with fully typed dataclasses that can be instantiated, compared for equality, and have sensible defaults. All models are documented and include from_dict/to_dict methods for serialization.

**Validation**:
- [ ] `SessionState`, `SessionMetadata`, and `CompactionState` dataclasses are defined in `src/claude_agent_sdk/models.py`
- [ ] All fields have proper type annotations (using `typing` module where needed)
- [ ] Dataclasses use `frozen=False` for mutability where needed
- [ ] Each dataclass has a docstring describing its purpose
- [ ] Unit tests in `tests/test_models.py` verify instantiation and field access
- [ ] `pytest tests/test_models.py -v` passes with 100% coverage of models.py

<prompt>
You are implementing the core data models for the Stateless Claude Agent SDK. These models represent session state and will be persisted/loaded by storage backends.

1. **Create `src/claude_agent_sdk/models.py`** with the following content:

```python
"""Core data models for session state management."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class PermissionMode(str, Enum):
    """Agent permission modes for tool execution."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ASK = "ask"


@dataclass
class CompactionState:
    """
    Tracks the state of context compaction for a session.

    When sessions grow too long, older messages are summarized
    to save context window space. This tracks what was compacted.
    """
    last_compaction_turn: int
    """Turn number when compaction last occurred"""

    summary: str
    """Text summary of the compacted messages"""

    original_message_ids: list[str]
    """IDs of messages that were replaced by the summary"""


@dataclass
class SessionMetadata:
    """
    Configuration and runtime state for an agent session.

    This includes model settings, permissions, and accumulated
    usage statistics.
    """
    model: str = "claude-sonnet-4-5-20250929"
    """Claude model to use for this session"""

    working_directory: str = "."
    """Working directory for file operations"""

    permission_mode: PermissionMode = PermissionMode.ASK
    """How to handle tool execution permissions"""

    allowed_tools: list[str] = field(default_factory=list)
    """Tools explicitly allowed (empty means all allowed)"""

    disallowed_tools: list[str] = field(default_factory=list)
    """Tools explicitly disallowed"""

    total_cost_usd: float = 0.0
    """Accumulated API cost for this session in USD"""

    num_turns: int = 0
    """Number of conversation turns completed"""

    usage: dict[str, int] = field(default_factory=dict)
    """Token usage statistics (input_tokens, output_tokens, etc.)"""

    compaction_state: CompactionState | None = None
    """Context compaction state, if compaction has occurred"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "model": self.model,
            "working_directory": self.working_directory,
            "permission_mode": self.permission_mode.value,
            "allowed_tools": self.allowed_tools,
            "disallowed_tools": self.disallowed_tools,
            "total_cost_usd": self.total_cost_usd,
            "num_turns": self.num_turns,
            "usage": self.usage,
            "compaction_state": (
                {
                    "last_compaction_turn": self.compaction_state.last_compaction_turn,
                    "summary": self.compaction_state.summary,
                    "original_message_ids": self.compaction_state.original_message_ids,
                }
                if self.compaction_state
                else None
            ),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionMetadata":
        """Create instance from dictionary."""
        compaction_data = data.get("compaction_state")
        compaction_state = (
            CompactionState(
                last_compaction_turn=compaction_data["last_compaction_turn"],
                summary=compaction_data["summary"],
                original_message_ids=compaction_data["original_message_ids"],
            )
            if compaction_data
            else None
        )

        return cls(
            model=data.get("model", "claude-sonnet-4-5-20250929"),
            working_directory=data.get("working_directory", "."),
            permission_mode=PermissionMode(data.get("permission_mode", "ask")),
            allowed_tools=data.get("allowed_tools", []),
            disallowed_tools=data.get("disallowed_tools", []),
            total_cost_usd=data.get("total_cost_usd", 0.0),
            num_turns=data.get("num_turns", 0),
            usage=data.get("usage", {}),
            compaction_state=compaction_state,
        )


@dataclass
class Message:
    """
    Base class for all message types in a conversation.

    This is a simplified placeholder - the real implementation
    would use the Anthropic SDK's message types.
    """
    role: str
    content: str | list[dict[str, Any]]
    """Message content (text or structured content blocks)"""

    id: str | None = None
    """Unique message identifier"""

    timestamp: datetime = field(default_factory=datetime.now)
    """When this message was created"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Message":
        """Create instance from dictionary."""
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id"),
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


@dataclass
class SessionState:
    """
    Complete snapshot of an agent session.

    This represents everything needed to resume or inspect
    a session: full message history, metadata, and timestamps.
    """
    session_id: str
    """Unique identifier for this session"""

    messages: list[Message]
    """Complete conversation history"""

    metadata: SessionMetadata
    """Session configuration and runtime state"""

    created_at: datetime
    """When this session was first created"""

    updated_at: datetime
    """When this session was last modified"""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "metadata": self.metadata.to_dict(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionState":
        """Create instance from dictionary."""
        return cls(
            session_id=data["session_id"],
            messages=[Message.from_dict(msg) for msg in data["messages"]],
            metadata=SessionMetadata.from_dict(data["metadata"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )
```

2. **Create comprehensive unit tests** in `tests/test_models.py`:

```python
"""Tests for core data models."""

import pytest
from datetime import datetime
from claude_agent_sdk.models import (
    SessionState,
    SessionMetadata,
    CompactionState,
    Message,
    PermissionMode,
)


def test_permission_mode_enum():
    """Test PermissionMode enum values."""
    assert PermissionMode.ENABLED.value == "enabled"
    assert PermissionMode.DISABLED.value == "disabled"
    assert PermissionMode.ASK.value == "ask"


def test_compaction_state_creation():
    """Test CompactionState instantiation."""
    state = CompactionState(
        last_compaction_turn=10,
        summary="Previous conversation about X",
        original_message_ids=["msg1", "msg2", "msg3"],
    )
    assert state.last_compaction_turn == 10
    assert "X" in state.summary
    assert len(state.original_message_ids) == 3


def test_session_metadata_defaults():
    """Test SessionMetadata has sensible defaults."""
    metadata = SessionMetadata()
    assert metadata.model == "claude-sonnet-4-5-20250929"
    assert metadata.working_directory == "."
    assert metadata.permission_mode == PermissionMode.ASK
    assert metadata.allowed_tools == []
    assert metadata.disallowed_tools == []
    assert metadata.total_cost_usd == 0.0
    assert metadata.num_turns == 0
    assert metadata.usage == {}
    assert metadata.compaction_state is None


def test_session_metadata_serialization():
    """Test SessionMetadata to_dict and from_dict."""
    original = SessionMetadata(
        model="claude-opus-4",
        working_directory="/tmp/test",
        permission_mode=PermissionMode.ENABLED,
        total_cost_usd=1.23,
        num_turns=5,
        usage={"input_tokens": 100, "output_tokens": 50},
    )

    # Serialize and deserialize
    data = original.to_dict()
    restored = SessionMetadata.from_dict(data)

    assert restored.model == original.model
    assert restored.working_directory == original.working_directory
    assert restored.permission_mode == original.permission_mode
    assert restored.total_cost_usd == original.total_cost_usd
    assert restored.num_turns == original.num_turns
    assert restored.usage == original.usage


def test_session_metadata_with_compaction():
    """Test SessionMetadata serialization with compaction state."""
    compaction = CompactionState(
        last_compaction_turn=15,
        summary="Early conversation summary",
        original_message_ids=["m1", "m2"],
    )
    metadata = SessionMetadata(compaction_state=compaction)

    data = metadata.to_dict()
    restored = SessionMetadata.from_dict(data)

    assert restored.compaction_state is not None
    assert restored.compaction_state.last_compaction_turn == 15
    assert "Early" in restored.compaction_state.summary


def test_message_creation():
    """Test Message instantiation."""
    msg = Message(
        role="user",
        content="Hello, Claude!",
        id="msg-123",
    )
    assert msg.role == "user"
    assert msg.content == "Hello, Claude!"
    assert msg.id == "msg-123"
    assert isinstance(msg.timestamp, datetime)


def test_message_serialization():
    """Test Message to_dict and from_dict."""
    original = Message(
        role="assistant",
        content=[{"type": "text", "text": "Hello!"}],
        id="msg-456",
    )

    data = original.to_dict()
    restored = Message.from_dict(data)

    assert restored.role == original.role
    assert restored.content == original.content
    assert restored.id == original.id


def test_session_state_creation():
    """Test SessionState instantiation."""
    now = datetime.now()
    messages = [
        Message(role="user", content="Hi"),
        Message(role="assistant", content="Hello!"),
    ]
    metadata = SessionMetadata(num_turns=1)

    state = SessionState(
        session_id="test-session",
        messages=messages,
        metadata=metadata,
        created_at=now,
        updated_at=now,
    )

    assert state.session_id == "test-session"
    assert len(state.messages) == 2
    assert state.metadata.num_turns == 1


def test_session_state_serialization():
    """Test SessionState to_dict and from_dict roundtrip."""
    now = datetime.now()
    messages = [Message(role="user", content="Test")]
    metadata = SessionMetadata(model="claude-sonnet-4-5-20250929")

    original = SessionState(
        session_id="serialize-test",
        messages=messages,
        metadata=metadata,
        created_at=now,
        updated_at=now,
    )

    data = original.to_dict()
    restored = SessionState.from_dict(data)

    assert restored.session_id == original.session_id
    assert len(restored.messages) == len(original.messages)
    assert restored.messages[0].content == original.messages[0].content
    assert restored.metadata.model == original.metadata.model
```

3. **Run the tests and verify coverage**:
   ```bash
   pytest tests/test_models.py -v --cov=src/claude_agent_sdk/models --cov-report=term-missing
   ```

4. **Update `src/claude_agent_sdk/__init__.py`** to export the models:
   ```python
   from claude_agent_sdk.models import (
       SessionState,
       SessionMetadata,
       CompactionState,
       Message,
       PermissionMode,
   )

   __all__ = [
       "SessionState",
       "SessionMetadata",
       "CompactionState",
       "Message",
       "PermissionMode",
   ]
   ```

Ensure all tests pass and you have high coverage of the models module.
</prompt>

---

#### Task 3: Define SessionStore Protocol

Status: **Pending**

**Goal**: Create the `SessionStore` Protocol that defines the interface for session state persistence, enabling pluggable storage backends while maintaining type safety and clear contracts.

**Working Result**: A `protocols.py` module containing the `SessionStore` Protocol with all required methods (create_session, load_session, append_message, etc.), complete with type annotations, docstrings, and custom exception classes.

**Validation**:
- [ ] `SessionStore` protocol is defined in `src/claude_agent_sdk/protocols.py`
- [ ] All 11 protocol methods are defined with correct signatures
- [ ] Custom exceptions (`SessionNotFoundError`, `StorageError`, `ConcurrencyError`) are defined
- [ ] Each method has a comprehensive docstring with parameters, return types, and raises
- [ ] `mypy src/claude_agent_sdk/protocols.py` passes with no errors
- [ ] Unit tests in `tests/test_protocols.py` verify protocol compliance checking

<prompt>
You are implementing the SessionStore Protocol that defines the contract for all storage backends in the Stateless Claude Agent SDK.

1. **Create `src/claude_agent_sdk/protocols.py`** with the following content:

```python
"""Protocol definitions for storage backends."""

from typing import Protocol, runtime_checkable
from claude_agent_sdk.models import SessionState, SessionMetadata, CompactionState, Message


# Custom exceptions for storage operations
class SessionNotFoundError(Exception):
    """Raised when attempting to access a session that doesn't exist."""
    pass


class StorageError(Exception):
    """Raised when a storage operation fails."""
    pass


class ConcurrencyError(Exception):
    """Raised when concurrent write conflicts are detected."""
    pass


@runtime_checkable
class SessionStore(Protocol):
    """
    Protocol defining the interface for session state persistence.

    All implementations must be thread-safe and handle concurrent access
    appropriately for their storage backend. Methods are async to support
    I/O-bound operations.

    Implementations should handle the following:
    - Atomic operations where possible
    - Proper error handling and recovery
    - Resource cleanup (connections, file handles, etc.)
    """

    async def create_session(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> None:
        """
        Initialize a new session with metadata.

        Args:
            session_id: Unique identifier for the session
            metadata: Initial session configuration and state

        Raises:
            StorageError: If session creation fails
        """
        ...

    async def load_session(
        self,
        session_id: str
    ) -> SessionState | None:
        """
        Load complete session state.

        Args:
            session_id: Unique identifier for the session

        Returns:
            Complete session state, or None if session doesn't exist

        Raises:
            StorageError: If loading fails due to corruption or I/O error
        """
        ...

    async def append_message(
        self,
        session_id: str,
        message: Message
    ) -> None:
        """
        Append a single message to session history.

        This operation must be atomic and preserve message order.

        Args:
            session_id: Session to append to
            message: Message to append

        Raises:
            SessionNotFoundError: If session doesn't exist
            StorageError: If persistence fails
            ConcurrencyError: If concurrent write detected
        """
        ...

    async def update_metadata(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> None:
        """
        Update session metadata (costs, usage, etc.).

        Args:
            session_id: Session to update
            metadata: New metadata to persist

        Raises:
            SessionNotFoundError: If session doesn't exist
            StorageError: If update fails
        """
        ...

    async def get_messages(
        self,
        session_id: str,
        from_turn: int = 0,
        to_turn: int | None = None
    ) -> list[Message]:
        """
        Retrieve message history with optional range.

        Useful for pagination and context windowing in large sessions.

        Args:
            session_id: Session to retrieve messages from
            from_turn: Starting turn number (inclusive), defaults to 0
            to_turn: Ending turn number (exclusive), defaults to end

        Returns:
            List of messages in the specified range

        Raises:
            SessionNotFoundError: If session doesn't exist
            StorageError: If retrieval fails
        """
        ...

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str
    ) -> None:
        """
        Create a copy of a session with a new ID.

        Used for session branching (creating alternate conversation paths).

        Args:
            source_session_id: Session to copy from
            new_session_id: ID for the new session copy

        Raises:
            SessionNotFoundError: If source session doesn't exist
            StorageError: If fork operation fails
        """
        ...

    async def compact_session(
        self,
        session_id: str,
        compaction_state: CompactionState
    ) -> None:
        """
        Apply context compaction to session.

        Replaces message ranges with summaries to save context window space.
        The compaction_state indicates which messages were summarized.

        Args:
            session_id: Session to compact
            compaction_state: Details of what was compacted

        Raises:
            SessionNotFoundError: If session doesn't exist
            StorageError: If compaction fails
        """
        ...

    async def list_sessions(
        self,
        working_directory: str | None = None,
        limit: int = 100
    ) -> list[str]:
        """
        List session IDs, optionally filtered by working directory.

        Used for --continue functionality and session discovery.

        Args:
            working_directory: Filter to sessions in this directory, or None for all
            limit: Maximum number of session IDs to return

        Returns:
            List of session IDs, most recently updated first

        Raises:
            StorageError: If listing fails
        """
        ...

    async def delete_session(
        self,
        session_id: str
    ) -> None:
        """
        Permanently delete a session.

        Args:
            session_id: Session to delete

        Raises:
            SessionNotFoundError: If session doesn't exist
            StorageError: If deletion fails
        """
        ...

    async def session_exists(
        self,
        session_id: str
    ) -> bool:
        """
        Check if session exists without loading full state.

        This is a lightweight operation for existence checking.

        Args:
            session_id: Session to check

        Returns:
            True if session exists, False otherwise

        Raises:
            StorageError: If check fails
        """
        ...

    async def close(self) -> None:
        """
        Clean up resources (connections, file handles, etc.).

        Should be called when the store is no longer needed.
        Implementations should be idempotent (safe to call multiple times).
        """
        ...
```

2. **Create protocol compliance tests** in `tests/test_protocols.py`:

```python
"""Tests for SessionStore protocol definition."""

import pytest
from claude_agent_sdk.protocols import (
    SessionStore,
    SessionNotFoundError,
    StorageError,
    ConcurrencyError,
)
from claude_agent_sdk.models import SessionMetadata, CompactionState, Message


def test_exceptions_exist():
    """Test that custom exceptions are defined."""
    assert issubclass(SessionNotFoundError, Exception)
    assert issubclass(StorageError, Exception)
    assert issubclass(ConcurrencyError, Exception)


def test_exceptions_can_be_raised():
    """Test that exceptions can be instantiated and raised."""
    with pytest.raises(SessionNotFoundError):
        raise SessionNotFoundError("Session not found")

    with pytest.raises(StorageError):
        raise StorageError("Storage failed")

    with pytest.raises(ConcurrencyError):
        raise ConcurrencyError("Concurrent write conflict")


def test_protocol_is_runtime_checkable():
    """Test that SessionStore protocol can check instances at runtime."""
    # This test verifies the @runtime_checkable decorator works
    from typing import runtime_checkable

    # SessionStore should be runtime checkable
    assert hasattr(SessionStore, "__protocol_attrs__") or hasattr(SessionStore, "_is_protocol")


class MockStore:
    """Mock implementation of SessionStore for testing."""

    async def create_session(self, session_id: str, metadata: SessionMetadata) -> None:
        pass

    async def load_session(self, session_id: str):
        return None

    async def append_message(self, session_id: str, message: Message) -> None:
        pass

    async def update_metadata(self, session_id: str, metadata: SessionMetadata) -> None:
        pass

    async def get_messages(self, session_id: str, from_turn: int = 0, to_turn: int | None = None):
        return []

    async def fork_session(self, source_session_id: str, new_session_id: str) -> None:
        pass

    async def compact_session(self, session_id: str, compaction_state: CompactionState) -> None:
        pass

    async def list_sessions(self, working_directory: str | None = None, limit: int = 100):
        return []

    async def delete_session(self, session_id: str) -> None:
        pass

    async def session_exists(self, session_id: str) -> bool:
        return False

    async def close(self) -> None:
        pass


def test_mock_store_satisfies_protocol():
    """Test that a complete mock implementation satisfies the protocol."""
    store = MockStore()
    # In Python 3.10+, runtime_checkable protocols support isinstance
    # This will pass if MockStore has all required methods
    assert isinstance(store, SessionStore)


class IncompleteStore:
    """Incomplete implementation missing some methods."""

    async def create_session(self, session_id: str, metadata: SessionMetadata) -> None:
        pass

    async def load_session(self, session_id: str):
        return None


def test_incomplete_store_fails_protocol_check():
    """Test that incomplete implementations don't satisfy the protocol."""
    store = IncompleteStore()
    # This should fail because IncompleteStore is missing most methods
    assert not isinstance(store, SessionStore)


@pytest.mark.asyncio
async def test_mock_store_methods_callable():
    """Test that mock store methods can be called."""
    store = MockStore()
    metadata = SessionMetadata()

    # These should all complete without error
    await store.create_session("test-id", metadata)
    result = await store.load_session("test-id")
    assert result is None

    exists = await store.session_exists("test-id")
    assert exists is False

    sessions = await store.list_sessions()
    assert sessions == []

    await store.close()
```

3. **Run mypy type checking**:
   ```bash
   mypy src/claude_agent_sdk/protocols.py --strict
   ```

4. **Run the protocol tests**:
   ```bash
   pytest tests/test_protocols.py -v
   ```

5. **Update `src/claude_agent_sdk/__init__.py`** to export the protocol:
   ```python
   from claude_agent_sdk.protocols import (
       SessionStore,
       SessionNotFoundError,
       StorageError,
       ConcurrencyError,
   )

   __all__ = [
       # ... existing exports ...
       "SessionStore",
       "SessionNotFoundError",
       "StorageError",
       "ConcurrencyError",
   ]
   ```

Ensure all tests pass and mypy type checking succeeds.
</prompt>

---

#### Task 4: Implement Message Serialization Utilities

Status: **Pending**

**Goal**: Create robust serialization/deserialization utilities that convert Message objects to/from JSON format, handling all content types (text, tool calls, tool results) and maintaining type safety.

**Working Result**: A `serialization.py` module with `serialize_message()` and `deserialize_message()` functions that handle all message types, preserve all fields, and raise clear errors for invalid data.

**Validation**:
- [ ] `serialize_message()` and `deserialize_message()` functions exist in `src/claude_agent_sdk/serialization.py`
- [ ] Functions handle text content, structured content (list of blocks), and None values
- [ ] Roundtrip serialization preserves all message fields exactly
- [ ] Invalid JSON or missing required fields raise appropriate exceptions
- [ ] Unit tests in `tests/test_serialization.py` cover all message types and edge cases
- [ ] `pytest tests/test_serialization.py -v` passes with 100% coverage

<prompt>
You are implementing message serialization utilities for the Stateless Claude Agent SDK. These utilities will be used by all storage backends to persist and load messages.

1. **Create `src/claude_agent_sdk/serialization.py`** with the following content:

```python
"""Utilities for serializing and deserializing messages."""

import json
from typing import Any
from datetime import datetime
from claude_agent_sdk.models import Message


class SerializationError(Exception):
    """Raised when serialization or deserialization fails."""
    pass


def serialize_message(message: Message) -> dict[str, Any]:
    """
    Convert a Message object to a JSON-serializable dictionary.

    Args:
        message: Message to serialize

    Returns:
        Dictionary suitable for JSON encoding

    Raises:
        SerializationError: If message cannot be serialized

    Examples:
        >>> msg = Message(role="user", content="Hello")
        >>> data = serialize_message(msg)
        >>> assert data["role"] == "user"
    """
    try:
        return {
            "role": message.role,
            "content": message.content,
            "id": message.id,
            "timestamp": message.timestamp.isoformat() if message.timestamp else None,
        }
    except Exception as e:
        raise SerializationError(f"Failed to serialize message: {e}") from e


def deserialize_message(data: dict[str, Any]) -> Message:
    """
    Convert a dictionary to a Message object.

    Args:
        data: Dictionary containing message fields

    Returns:
        Reconstructed Message object

    Raises:
        SerializationError: If data is invalid or missing required fields

    Examples:
        >>> data = {"role": "user", "content": "Hello", "id": "123", "timestamp": "2024-01-01T00:00:00"}
        >>> msg = deserialize_message(data)
        >>> assert msg.role == "user"
    """
    try:
        # Validate required fields
        if "role" not in data:
            raise SerializationError("Missing required field: role")
        if "content" not in data:
            raise SerializationError("Missing required field: content")

        # Parse timestamp
        timestamp = None
        if data.get("timestamp"):
            try:
                timestamp = datetime.fromisoformat(data["timestamp"])
            except (ValueError, TypeError) as e:
                raise SerializationError(f"Invalid timestamp format: {data['timestamp']}") from e

        return Message(
            role=data["role"],
            content=data["content"],
            id=data.get("id"),
            timestamp=timestamp or datetime.now(),
        )
    except SerializationError:
        raise
    except Exception as e:
        raise SerializationError(f"Failed to deserialize message: {e}") from e


def serialize_message_to_json(message: Message) -> str:
    """
    Serialize a message directly to a JSON string.

    Args:
        message: Message to serialize

    Returns:
        JSON string representation

    Raises:
        SerializationError: If serialization fails
    """
    try:
        data = serialize_message(message)
        return json.dumps(data, ensure_ascii=False)
    except json.JSONEncodeError as e:
        raise SerializationError(f"JSON encoding failed: {e}") from e


def deserialize_message_from_json(json_str: str) -> Message:
    """
    Deserialize a message directly from a JSON string.

    Args:
        json_str: JSON string containing message data

    Returns:
        Reconstructed Message object

    Raises:
        SerializationError: If deserialization fails
    """
    try:
        data = json.loads(json_str)
        return deserialize_message(data)
    except json.JSONDecodeError as e:
        raise SerializationError(f"JSON decoding failed: {e}") from e


def serialize_messages_batch(messages: list[Message]) -> list[dict[str, Any]]:
    """
    Serialize multiple messages at once.

    Args:
        messages: List of messages to serialize

    Returns:
        List of serialized message dictionaries

    Raises:
        SerializationError: If any message fails to serialize
    """
    return [serialize_message(msg) for msg in messages]


def deserialize_messages_batch(data_list: list[dict[str, Any]]) -> list[Message]:
    """
    Deserialize multiple messages at once.

    Args:
        data_list: List of message data dictionaries

    Returns:
        List of reconstructed Message objects

    Raises:
        SerializationError: If any message fails to deserialize
    """
    return [deserialize_message(data) for data in data_list]
```

2. **Create comprehensive unit tests** in `tests/test_serialization.py`:

```python
"""Tests for message serialization utilities."""

import pytest
import json
from datetime import datetime
from claude_agent_sdk.serialization import (
    serialize_message,
    deserialize_message,
    serialize_message_to_json,
    deserialize_message_from_json,
    serialize_messages_batch,
    deserialize_messages_batch,
    SerializationError,
)
from claude_agent_sdk.models import Message


def test_serialize_simple_text_message():
    """Test serializing a message with simple text content."""
    msg = Message(
        role="user",
        content="Hello, world!",
        id="msg-123",
    )

    data = serialize_message(msg)

    assert data["role"] == "user"
    assert data["content"] == "Hello, world!"
    assert data["id"] == "msg-123"
    assert "timestamp" in data


def test_serialize_structured_content():
    """Test serializing a message with structured content blocks."""
    msg = Message(
        role="assistant",
        content=[
            {"type": "text", "text": "Here's the result:"},
            {"type": "tool_use", "id": "tool-1", "name": "calculator"},
        ],
        id="msg-456",
    )

    data = serialize_message(msg)

    assert data["role"] == "assistant"
    assert isinstance(data["content"], list)
    assert len(data["content"]) == 2
    assert data["content"][0]["type"] == "text"


def test_serialize_message_without_id():
    """Test serializing a message without an ID."""
    msg = Message(role="user", content="Test")
    data = serialize_message(msg)

    assert data["role"] == "user"
    assert data["id"] is None


def test_deserialize_simple_message():
    """Test deserializing a simple message."""
    data = {
        "role": "user",
        "content": "Hello",
        "id": "msg-789",
        "timestamp": "2024-01-15T10:30:00",
    }

    msg = deserialize_message(data)

    assert msg.role == "user"
    assert msg.content == "Hello"
    assert msg.id == "msg-789"
    assert isinstance(msg.timestamp, datetime)


def test_deserialize_structured_content():
    """Test deserializing a message with structured content."""
    data = {
        "role": "assistant",
        "content": [{"type": "text", "text": "Result"}],
        "id": "msg-999",
        "timestamp": "2024-01-15T10:30:00",
    }

    msg = deserialize_message(data)

    assert msg.role == "assistant"
    assert isinstance(msg.content, list)
    assert msg.content[0]["type"] == "text"


def test_deserialize_missing_role():
    """Test that deserializing without role raises error."""
    data = {"content": "Hello"}

    with pytest.raises(SerializationError, match="Missing required field: role"):
        deserialize_message(data)


def test_deserialize_missing_content():
    """Test that deserializing without content raises error."""
    data = {"role": "user"}

    with pytest.raises(SerializationError, match="Missing required field: content"):
        deserialize_message(data)


def test_deserialize_invalid_timestamp():
    """Test that invalid timestamp format raises error."""
    data = {
        "role": "user",
        "content": "Hello",
        "timestamp": "not-a-timestamp",
    }

    with pytest.raises(SerializationError, match="Invalid timestamp format"):
        deserialize_message(data)


def test_roundtrip_serialization():
    """Test that serialize -> deserialize preserves all fields."""
    original = Message(
        role="assistant",
        content=[{"type": "text", "text": "Roundtrip test"}],
        id="msg-roundtrip",
        timestamp=datetime(2024, 1, 15, 12, 0, 0),
    )

    # Serialize and deserialize
    data = serialize_message(original)
    restored = deserialize_message(data)

    assert restored.role == original.role
    assert restored.content == original.content
    assert restored.id == original.id
    # Timestamps should be equal (within serialization precision)
    assert restored.timestamp.date() == original.timestamp.date()


def test_serialize_to_json_string():
    """Test direct serialization to JSON string."""
    msg = Message(role="user", content="JSON test", id="json-1")

    json_str = serialize_message_to_json(msg)

    assert isinstance(json_str, str)
    # Verify it's valid JSON
    data = json.loads(json_str)
    assert data["role"] == "user"


def test_deserialize_from_json_string():
    """Test direct deserialization from JSON string."""
    json_str = '{"role": "assistant", "content": "Test", "id": "json-2", "timestamp": "2024-01-15T10:00:00"}'

    msg = deserialize_message_from_json(json_str)

    assert msg.role == "assistant"
    assert msg.content == "Test"


def test_deserialize_invalid_json():
    """Test that invalid JSON raises SerializationError."""
    invalid_json = "{not valid json}"

    with pytest.raises(SerializationError, match="JSON decoding failed"):
        deserialize_message_from_json(invalid_json)


def test_serialize_batch():
    """Test batch serialization of multiple messages."""
    messages = [
        Message(role="user", content="Message 1", id="m1"),
        Message(role="assistant", content="Message 2", id="m2"),
        Message(role="user", content="Message 3", id="m3"),
    ]

    data_list = serialize_messages_batch(messages)

    assert len(data_list) == 3
    assert data_list[0]["id"] == "m1"
    assert data_list[1]["id"] == "m2"
    assert data_list[2]["id"] == "m3"


def test_deserialize_batch():
    """Test batch deserialization of multiple messages."""
    data_list = [
        {"role": "user", "content": "Msg 1", "id": "m1", "timestamp": "2024-01-15T10:00:00"},
        {"role": "assistant", "content": "Msg 2", "id": "m2", "timestamp": "2024-01-15T10:01:00"},
    ]

    messages = deserialize_messages_batch(data_list)

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


def test_batch_roundtrip():
    """Test that batch serialize -> deserialize preserves all messages."""
    original_messages = [
        Message(role="user", content="First", id="1"),
        Message(role="assistant", content="Second", id="2"),
    ]

    data_list = serialize_messages_batch(original_messages)
    restored_messages = deserialize_messages_batch(data_list)

    assert len(restored_messages) == len(original_messages)
    for orig, restored in zip(original_messages, restored_messages):
        assert restored.role == orig.role
        assert restored.content == orig.content
        assert restored.id == orig.id


def test_empty_batch_serialization():
    """Test serializing an empty list of messages."""
    data_list = serialize_messages_batch([])
    assert data_list == []


def test_empty_batch_deserialization():
    """Test deserializing an empty list of messages."""
    messages = deserialize_messages_batch([])
    assert messages == []
```

3. **Run the tests with coverage**:
   ```bash
   pytest tests/test_serialization.py -v --cov=src/claude_agent_sdk/serialization --cov-report=term-missing
   ```

4. **Update `src/claude_agent_sdk/__init__.py`** to export serialization utilities:
   ```python
   from claude_agent_sdk.serialization import (
       serialize_message,
       deserialize_message,
       SerializationError,
   )

   __all__ = [
       # ... existing exports ...
       "serialize_message",
       "deserialize_message",
       "SerializationError",
   ]
   ```

Ensure all tests pass with 100% coverage of the serialization module.
</prompt>

---

### 🔄 **Iteration 2: File-Based Storage Implementation**

This iteration implements the FileSessionStore, which provides file-based persistence using JSONL format. This is the default storage backend and must maintain 100% backward compatibility with existing SDK behavior.

---

#### Task 5: Implement FileSessionStore Basic Operations

Status: **Pending**

**Goal**: Implement the core FileSessionStore class with session creation, message appending, and loading functionality using JSONL format with proper file locking for concurrent access safety.

**Working Result**: A working **FileSessionStore** class in `src/claude_agent_sdk/stores/file_store.py` that can create sessions, append messages atomically to JSONL files, and load complete session state from disk. All operations are thread-safe using asyncio locks.

**Validation**:
- [ ] `FileSessionStore` class exists in `src/claude_agent_sdk/stores/file_store.py`
- [ ] `__init__` accepts `base_path` parameter (defaults to `~/.claude`)
- [ ] `create_session()` creates directory structure and writes initial metadata
- [ ] `append_message()` appends single JSON line atomically to `.jsonl` file
- [ ] `load_session()` reads JSONL file and reconstructs SessionState
- [ ] `session_exists()` checks if session file exists without loading it
- [ ] Unit tests in `tests/stores/test_file_store.py` verify basic operations
- [ ] `pytest tests/stores/test_file_store.py::test_create_and_load_session -v` passes

<prompt>
You are implementing the FileSessionStore, the default storage backend for the Stateless Claude Agent SDK. This store uses JSONL (JSON Lines) format where each message is a single line in a file.

1. **Create `src/claude_agent_sdk/stores/__init__.py`** to make it a package:
   ```python
   """Storage backend implementations."""

   from claude_agent_sdk.stores.file_store import FileSessionStore

   __all__ = ["FileSessionStore"]
   ```

2. **Create `src/claude_agent_sdk/stores/file_store.py`** with the following implementation:

```python
"""File-based session storage using JSONL format."""

import json
import asyncio
import aiofiles
from pathlib import Path
from datetime import datetime
from typing import Dict
from claude_agent_sdk.protocols import (
    SessionStore,
    SessionNotFoundError,
    StorageError,
)
from claude_agent_sdk.models import (
    SessionState,
    SessionMetadata,
    Message,
    CompactionState,
)
from claude_agent_sdk.serialization import (
    serialize_message,
    deserialize_message,
)


class FileSessionStore:
    """
    File-based storage using JSONL format.

    100% compatible with existing SDK behavior. Each session is stored
    as a .jsonl file with one message per line. Session metadata is stored
    as a special line in the file.

    Thread-safe using asyncio locks per session.
    """

    def __init__(self, base_path: Path | str | None = None):
        """
        Initialize file store.

        Args:
            base_path: Root directory for session storage.
                      Defaults to ~/.claude
        """
        if base_path is None:
            base_path = Path.home() / ".claude"
        self.base_path = Path(base_path)
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_session_path(self, session_id: str) -> Path:
        """
        Convert session ID to file path.

        Encodes session ID to be filesystem-safe and maintains
        compatibility with existing path structure.
        """
        # Encode session ID for filesystem (replace problematic chars)
        encoded = session_id.replace("/", "-").replace("\\", "-")
        # Store in projects subdirectory
        session_dir = self.base_path / "projects" / encoded
        return session_dir / f"{encoded}.jsonl"

    def _get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a lock for a session."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    async def create_session(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> None:
        """
        Initialize a new session with metadata.

        Creates the session directory and writes initial metadata line.
        """
        path = self._get_session_path(session_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        lock = self._get_lock(session_id)

        async with lock:
            try:
                # Write metadata as first line
                metadata_entry = {
                    "type": "metadata",
                    "data": metadata.to_dict(),
                    "timestamp": datetime.now().isoformat(),
                }
                line = json.dumps(metadata_entry, ensure_ascii=False) + "\n"

                async with aiofiles.open(path, "w", encoding="utf-8") as f:
                    await f.write(line)
            except Exception as e:
                raise StorageError(f"Failed to create session {session_id}: {e}") from e

    async def append_message(
        self,
        session_id: str,
        message: Message
    ) -> None:
        """
        Append a single message to session history.

        Atomically appends one JSON line to the JSONL file.
        """
        path = self._get_session_path(session_id)

        if not path.exists():
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        lock = self._get_lock(session_id)

        async with lock:
            try:
                # Serialize message
                msg_data = serialize_message(message)
                msg_entry = {
                    "type": "message",
                    "data": msg_data,
                }
                line = json.dumps(msg_entry, ensure_ascii=False) + "\n"

                # Atomic append
                async with aiofiles.open(path, "a", encoding="utf-8") as f:
                    await f.write(line)
            except SessionNotFoundError:
                raise
            except Exception as e:
                raise StorageError(f"Failed to append message to {session_id}: {e}") from e

    async def load_session(
        self,
        session_id: str
    ) -> SessionState | None:
        """
        Load complete session state.

        Parses the JSONL file to reconstruct full session state.
        """
        path = self._get_session_path(session_id)

        if not path.exists():
            return None

        lock = self._get_lock(session_id)

        async with lock:
            try:
                messages = []
                metadata = None

                async with aiofiles.open(path, "r", encoding="utf-8") as f:
                    async for line in f:
                        line = line.strip()
                        if not line:
                            continue

                        entry = json.loads(line)

                        if entry.get("type") == "metadata":
                            metadata = SessionMetadata.from_dict(entry["data"])
                        elif entry.get("type") == "message":
                            msg = deserialize_message(entry["data"])
                            messages.append(msg)

                # Get file timestamps
                stat = path.stat()
                created_at = datetime.fromtimestamp(stat.st_ctime)
                updated_at = datetime.fromtimestamp(stat.st_mtime)

                # Use default metadata if none found
                if metadata is None:
                    metadata = SessionMetadata()

                return SessionState(
                    session_id=session_id,
                    messages=messages,
                    metadata=metadata,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            except Exception as e:
                raise StorageError(f"Failed to load session {session_id}: {e}") from e

    async def session_exists(
        self,
        session_id: str
    ) -> bool:
        """
        Check if session exists without loading full state.

        Lightweight existence check.
        """
        path = self._get_session_path(session_id)
        return path.exists()

    async def update_metadata(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> None:
        """
        Update session metadata.

        Rewrites the metadata line in the JSONL file.
        For file store, we need to rewrite the entire file.
        """
        # Load current session
        session = await self.load_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        # Update metadata
        session.metadata = metadata

        # Rewrite file with new metadata
        path = self._get_session_path(session_id)
        lock = self._get_lock(session_id)

        async with lock:
            try:
                # Write to temporary file first
                temp_path = path.with_suffix(".jsonl.tmp")

                async with aiofiles.open(temp_path, "w", encoding="utf-8") as f:
                    # Write metadata
                    metadata_entry = {
                        "type": "metadata",
                        "data": metadata.to_dict(),
                        "timestamp": datetime.now().isoformat(),
                    }
                    await f.write(json.dumps(metadata_entry, ensure_ascii=False) + "\n")

                    # Write all messages
                    for msg in session.messages:
                        msg_entry = {
                            "type": "message",
                            "data": serialize_message(msg),
                        }
                        await f.write(json.dumps(msg_entry, ensure_ascii=False) + "\n")

                # Atomic replace
                temp_path.replace(path)
            except Exception as e:
                raise StorageError(f"Failed to update metadata for {session_id}: {e}") from e

    async def close(self) -> None:
        """Clean up resources."""
        # Clear locks
        self._locks.clear()
```

3. **Create unit tests** in `tests/stores/test_file_store.py`:

```python
"""Tests for FileSessionStore."""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime
from claude_agent_sdk.stores.file_store import FileSessionStore
from claude_agent_sdk.models import SessionMetadata, Message
from claude_agent_sdk.protocols import SessionNotFoundError, StorageError


@pytest.fixture
def temp_store_path(tmp_path):
    """Create a temporary directory for file store tests."""
    return tmp_path / "test_store"


@pytest.fixture
def file_store(temp_store_path):
    """Create a FileSessionStore instance for testing."""
    return FileSessionStore(base_path=temp_store_path)


@pytest.mark.asyncio
async def test_create_session(file_store, temp_store_path):
    """Test creating a new session."""
    session_id = "test-session-1"
    metadata = SessionMetadata(model="claude-sonnet-4-5-20250929")

    await file_store.create_session(session_id, metadata)

    # Verify file was created
    session_path = file_store._get_session_path(session_id)
    assert session_path.exists()

    # Verify directory structure
    assert session_path.parent.exists()
    assert "test-session-1" in str(session_path)


@pytest.mark.asyncio
async def test_session_exists(file_store):
    """Test checking if a session exists."""
    session_id = "exists-test"
    metadata = SessionMetadata()

    # Should not exist initially
    assert not await file_store.session_exists(session_id)

    # Create session
    await file_store.create_session(session_id, metadata)

    # Should exist now
    assert await file_store.session_exists(session_id)


@pytest.mark.asyncio
async def test_append_message(file_store):
    """Test appending a message to a session."""
    session_id = "append-test"
    metadata = SessionMetadata()

    # Create session
    await file_store.create_session(session_id, metadata)

    # Append message
    msg = Message(role="user", content="Hello!", id="msg-1")
    await file_store.append_message(session_id, msg)

    # Verify message was appended
    session_path = file_store._get_session_path(session_id)
    content = session_path.read_text()
    assert "Hello!" in content
    assert "msg-1" in content


@pytest.mark.asyncio
async def test_append_message_to_nonexistent_session(file_store):
    """Test that appending to nonexistent session raises error."""
    msg = Message(role="user", content="Test")

    with pytest.raises(SessionNotFoundError):
        await file_store.append_message("nonexistent", msg)


@pytest.mark.asyncio
async def test_load_session(file_store):
    """Test loading a complete session."""
    session_id = "load-test"
    metadata = SessionMetadata(model="claude-opus-4", num_turns=2)

    # Create session
    await file_store.create_session(session_id, metadata)

    # Append messages
    msg1 = Message(role="user", content="First message", id="m1")
    msg2 = Message(role="assistant", content="Second message", id="m2")
    await file_store.append_message(session_id, msg1)
    await file_store.append_message(session_id, msg2)

    # Load session
    session = await file_store.load_session(session_id)

    assert session is not None
    assert session.session_id == session_id
    assert len(session.messages) == 2
    assert session.messages[0].content == "First message"
    assert session.messages[1].content == "Second message"
    assert session.metadata.model == "claude-opus-4"
    assert session.metadata.num_turns == 2


@pytest.mark.asyncio
async def test_load_nonexistent_session(file_store):
    """Test that loading nonexistent session returns None."""
    session = await file_store.load_session("does-not-exist")
    assert session is None


@pytest.mark.asyncio
async def test_update_metadata(file_store):
    """Test updating session metadata."""
    session_id = "metadata-test"
    metadata = SessionMetadata(num_turns=0, total_cost_usd=0.0)

    # Create session
    await file_store.create_session(session_id, metadata)

    # Append a message
    await file_store.append_message(
        session_id,
        Message(role="user", content="Test", id="m1")
    )

    # Update metadata
    updated_metadata = SessionMetadata(num_turns=1, total_cost_usd=0.05)
    await file_store.update_metadata(session_id, updated_metadata)

    # Load and verify
    session = await file_store.load_session(session_id)
    assert session.metadata.num_turns == 1
    assert session.metadata.total_cost_usd == 0.05
    # Message should still be there
    assert len(session.messages) == 1


@pytest.mark.asyncio
async def test_concurrent_appends(file_store):
    """Test that concurrent message appends are thread-safe."""
    session_id = "concurrent-test"
    metadata = SessionMetadata()

    await file_store.create_session(session_id, metadata)

    # Append multiple messages concurrently
    messages = [
        Message(role="user", content=f"Message {i}", id=f"msg-{i}")
        for i in range(10)
    ]

    await asyncio.gather(
        *[file_store.append_message(session_id, msg) for msg in messages]
    )

    # Load and verify all messages were saved
    session = await file_store.load_session(session_id)
    assert len(session.messages) == 10


@pytest.mark.asyncio
async def test_close(file_store):
    """Test cleanup of resources."""
    # Create some sessions
    await file_store.create_session("session1", SessionMetadata())
    await file_store.create_session("session2", SessionMetadata())

    # Close should clear locks
    await file_store.close()
    assert len(file_store._locks) == 0
```

4. **Run the tests**:
   ```bash
   pytest tests/stores/test_file_store.py -v --cov=src/claude_agent_sdk/stores/file_store
   ```

5. **Update `src/claude_agent_sdk/__init__.py`** to export FileSessionStore:
   ```python
   from claude_agent_sdk.stores import FileSessionStore

   __all__ = [
       # ... existing exports ...
       "FileSessionStore",
   ]
   ```

Ensure all tests pass and the file store correctly implements the SessionStore protocol.
</prompt>

---

#### Task 6: Implement FileSessionStore Advanced Operations

Status: **Pending**

**Goal**: Complete the FileSessionStore implementation by adding advanced operations: session listing, deletion, forking, and context compaction support.

**Working Result**: A fully functional FileSessionStore with all SessionStore protocol methods implemented, including `list_sessions()`, `delete_session()`, `fork_session()`, `compact_session()`, and `get_messages()` with range support.

**Validation**:
- [ ] `list_sessions()` lists all session IDs, optionally filtered by working directory
- [ ] `delete_session()` removes session file and cleans up directory if empty
- [ ] `fork_session()` creates a complete copy of a session with new ID
- [ ] `compact_session()` applies compaction and updates metadata
- [ ] `get_messages()` returns message subrange based on turn numbers
- [ ] All methods raise appropriate exceptions for error cases
- [ ] Unit tests in `tests/stores/test_file_store.py` cover all new operations
- [ ] `pytest tests/stores/test_file_store.py -v` passes with >95% coverage

<prompt>
You are completing the FileSessionStore implementation by adding advanced operations for session management and context windowing.

1. **Add the following methods to `src/claude_agent_sdk/stores/file_store.py`**:

```python
    async def get_messages(
        self,
        session_id: str,
        from_turn: int = 0,
        to_turn: int | None = None
    ) -> list[Message]:
        """
        Retrieve message history with optional range.

        Useful for pagination and context windowing.
        """
        session = await self.load_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        messages = session.messages

        if to_turn is None:
            return messages[from_turn:]
        else:
            return messages[from_turn:to_turn]

    async def list_sessions(
        self,
        working_directory: str | None = None,
        limit: int = 100
    ) -> list[str]:
        """
        List session IDs, optionally filtered by working directory.

        Returns most recently modified sessions first.
        """
        try:
            projects_dir = self.base_path / "projects"
            if not projects_dir.exists():
                return []

            # Find all .jsonl files
            session_files = []
            for path in projects_dir.rglob("*.jsonl"):
                # Extract session ID from filename
                session_id = path.stem  # filename without .jsonl extension

                # Filter by working directory if specified
                if working_directory is not None:
                    # Load session to check working directory
                    session = await self.load_session(session_id)
                    if session and session.metadata.working_directory == working_directory:
                        session_files.append((path.stat().st_mtime, session_id))
                else:
                    session_files.append((path.stat().st_mtime, session_id))

            # Sort by modification time (most recent first)
            session_files.sort(reverse=True, key=lambda x: x[0])

            # Return session IDs only, up to limit
            return [session_id for _, session_id in session_files[:limit]]
        except Exception as e:
            raise StorageError(f"Failed to list sessions: {e}") from e

    async def delete_session(
        self,
        session_id: str
    ) -> None:
        """
        Permanently delete a session.

        Removes the session file and cleans up empty directories.
        """
        path = self._get_session_path(session_id)

        if not path.exists():
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        lock = self._get_lock(session_id)

        async with lock:
            try:
                # Delete the file
                path.unlink()

                # Clean up empty parent directory
                parent = path.parent
                if parent.exists() and not any(parent.iterdir()):
                    parent.rmdir()

                # Remove lock
                if session_id in self._locks:
                    del self._locks[session_id]
            except Exception as e:
                raise StorageError(f"Failed to delete session {session_id}: {e}") from e

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str
    ) -> None:
        """
        Create a copy of a session with a new ID.

        Used for session branching.
        """
        # Load source session
        source_session = await self.load_session(source_session_id)
        if source_session is None:
            raise SessionNotFoundError(f"Source session {source_session_id} does not exist")

        # Create new session with same metadata
        await self.create_session(new_session_id, source_session.metadata)

        # Copy all messages
        for message in source_session.messages:
            await self.append_message(new_session_id, message)

    async def compact_session(
        self,
        session_id: str,
        compaction_state: CompactionState
    ) -> None:
        """
        Apply context compaction to session.

        Replaces compacted messages with a summary and updates metadata.
        This is a simplified implementation - real compaction would
        remove specific message ranges.
        """
        session = await self.load_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        # Update metadata with compaction state
        session.metadata.compaction_state = compaction_state

        # Update the session metadata
        await self.update_metadata(session_id, session.metadata)
```

2. **Add comprehensive tests for advanced operations** in `tests/stores/test_file_store.py`:

```python
@pytest.mark.asyncio
async def test_get_messages_full_range(file_store):
    """Test getting all messages from a session."""
    session_id = "messages-test"
    await file_store.create_session(session_id, SessionMetadata())

    # Add several messages
    for i in range(5):
        await file_store.append_message(
            session_id,
            Message(role="user", content=f"Message {i}", id=f"m{i}")
        )

    messages = await file_store.get_messages(session_id)
    assert len(messages) == 5
    assert messages[0].content == "Message 0"
    assert messages[4].content == "Message 4"


@pytest.mark.asyncio
async def test_get_messages_with_range(file_store):
    """Test getting a subset of messages."""
    session_id = "range-test"
    await file_store.create_session(session_id, SessionMetadata())

    for i in range(10):
        await file_store.append_message(
            session_id,
            Message(role="user", content=f"Msg {i}", id=f"m{i}")
        )

    # Get messages 3-7
    messages = await file_store.get_messages(session_id, from_turn=3, to_turn=7)
    assert len(messages) == 4
    assert messages[0].content == "Msg 3"
    assert messages[3].content == "Msg 6"


@pytest.mark.asyncio
async def test_get_messages_from_turn(file_store):
    """Test getting messages from a specific turn onwards."""
    session_id = "from-turn-test"
    await file_store.create_session(session_id, SessionMetadata())

    for i in range(5):
        await file_store.append_message(
            session_id,
            Message(role="user", content=f"Msg {i}", id=f"m{i}")
        )

    messages = await file_store.get_messages(session_id, from_turn=3)
    assert len(messages) == 2
    assert messages[0].content == "Msg 3"


@pytest.mark.asyncio
async def test_list_sessions(file_store):
    """Test listing all sessions."""
    # Create multiple sessions
    for i in range(3):
        await file_store.create_session(f"session-{i}", SessionMetadata())

    sessions = await file_store.list_sessions()
    assert len(sessions) == 3
    assert "session-0" in sessions
    assert "session-1" in sessions
    assert "session-2" in sessions


@pytest.mark.asyncio
async def test_list_sessions_empty(file_store):
    """Test listing when no sessions exist."""
    sessions = await file_store.list_sessions()
    assert sessions == []


@pytest.mark.asyncio
async def test_list_sessions_with_limit(file_store):
    """Test listing sessions with a limit."""
    # Create many sessions
    for i in range(10):
        await file_store.create_session(f"session-{i}", SessionMetadata())
        await asyncio.sleep(0.01)  # Ensure different mtimes

    sessions = await file_store.list_sessions(limit=5)
    assert len(sessions) == 5


@pytest.mark.asyncio
async def test_list_sessions_by_working_directory(file_store):
    """Test filtering sessions by working directory."""
    # Create sessions in different directories
    metadata1 = SessionMetadata(working_directory="/tmp/project1")
    metadata2 = SessionMetadata(working_directory="/tmp/project2")

    await file_store.create_session("session-1", metadata1)
    await file_store.create_session("session-2", metadata2)
    await file_store.create_session("session-3", metadata1)

    # List sessions in project1
    sessions = await file_store.list_sessions(working_directory="/tmp/project1")
    assert len(sessions) == 2
    assert "session-1" in sessions
    assert "session-3" in sessions


@pytest.mark.asyncio
async def test_delete_session(file_store):
    """Test deleting a session."""
    session_id = "delete-test"
    await file_store.create_session(session_id, SessionMetadata())

    # Verify exists
    assert await file_store.session_exists(session_id)

    # Delete
    await file_store.delete_session(session_id)

    # Verify gone
    assert not await file_store.session_exists(session_id)


@pytest.mark.asyncio
async def test_delete_nonexistent_session(file_store):
    """Test that deleting nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        await file_store.delete_session("does-not-exist")


@pytest.mark.asyncio
async def test_fork_session(file_store):
    """Test forking a session to create a copy."""
    source_id = "source-session"
    fork_id = "forked-session"

    # Create source session with messages
    metadata = SessionMetadata(model="claude-opus-4", num_turns=2)
    await file_store.create_session(source_id, metadata)
    await file_store.append_message(
        source_id,
        Message(role="user", content="Original message", id="m1")
    )

    # Fork the session
    await file_store.fork_session(source_id, fork_id)

    # Verify fork exists and has same content
    fork_session = await file_store.load_session(fork_id)
    assert fork_session is not None
    assert fork_session.session_id == fork_id
    assert len(fork_session.messages) == 1
    assert fork_session.messages[0].content == "Original message"
    assert fork_session.metadata.model == "claude-opus-4"

    # Verify source still exists unchanged
    source_session = await file_store.load_session(source_id)
    assert source_session is not None


@pytest.mark.asyncio
async def test_fork_nonexistent_session(file_store):
    """Test that forking nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        await file_store.fork_session("nonexistent", "new-fork")


@pytest.mark.asyncio
async def test_compact_session(file_store):
    """Test applying context compaction to a session."""
    from claude_agent_sdk.models import CompactionState

    session_id = "compact-test"
    metadata = SessionMetadata()
    await file_store.create_session(session_id, metadata)

    # Add messages
    for i in range(5):
        await file_store.append_message(
            session_id,
            Message(role="user", content=f"Msg {i}", id=f"m{i}")
        )

    # Apply compaction
    compaction = CompactionState(
        last_compaction_turn=3,
        summary="Summary of messages 0-2",
        original_message_ids=["m0", "m1", "m2"]
    )
    await file_store.compact_session(session_id, compaction)

    # Verify compaction state was saved
    session = await file_store.load_session(session_id)
    assert session.metadata.compaction_state is not None
    assert session.metadata.compaction_state.last_compaction_turn == 3
    assert "Summary" in session.metadata.compaction_state.summary


@pytest.mark.asyncio
async def test_protocol_compliance(file_store):
    """Test that FileSessionStore satisfies SessionStore protocol."""
    from claude_agent_sdk.protocols import SessionStore

    # FileSessionStore should satisfy the protocol
    assert isinstance(file_store, SessionStore)
```

3. **Run all file store tests**:
   ```bash
   pytest tests/stores/test_file_store.py -v --cov=src/claude_agent_sdk/stores/file_store --cov-report=term-missing
   ```

4. **Verify protocol compliance**:
   ```bash
   mypy src/claude_agent_sdk/stores/file_store.py --strict
   ```

Ensure all tests pass with high coverage and the FileSessionStore fully implements the SessionStore protocol.
</prompt>

---

### 🔄 **Iteration 3: Redis Storage Backend**

This iteration implements the RedisSessionStore for distributed and stateless deployments. Redis provides fast, in-memory storage with built-in TTL support, perfect for horizontally-scaled agent systems.

---

#### Task 7: Implement RedisSessionStore Basic Operations

Status: **Pending**

**Goal**: Implement a Redis-based storage backend using Redis lists for messages and hashes for metadata, supporting distributed agent deployments with automatic session expiration.

**Working Result**: A working **RedisSessionStore** class in `src/claude_agent_sdk/stores/redis_store.py` that stores messages in Redis lists, metadata in Redis hashes, and supports configurable TTL for automatic session cleanup.

**Validation**:
- [ ] `RedisSessionStore` class exists in `src/claude_agent_sdk/stores/redis_store.py`
- [ ] `__init__` accepts `redis_url` parameter (defaults to `redis://localhost:6379`)
- [ ] Uses Redis data structures: `session:{id}:messages` (list), `session:{id}:metadata` (hash)
- [ ] `create_session()` initializes metadata hash and created_at timestamp
- [ ] `append_message()` uses RPUSH to append to message list atomically
- [ ] `load_session()` loads messages from list and metadata from hash
- [ ] All session keys have configurable TTL (default 30 days)
- [ ] Unit tests use fakeredis for testing without actual Redis server
- [ ] `pytest tests/stores/test_redis_store.py -v` passes

<prompt>
You are implementing the RedisSessionStore for distributed, stateless agent deployments. This store uses Redis as a fast, shared storage backend.

1. **Install redis dependency** - Ensure `redis[hiredis]>=5.0.0` is in pyproject.toml dependencies.

2. **Create `src/claude_agent_sdk/stores/redis_store.py`** with the following implementation:

```python
"""Redis-based session storage for distributed deployments."""

import json
from datetime import datetime, timedelta
from typing import Optional
import redis.asyncio as redis
from claude_agent_sdk.protocols import (
    SessionStore,
    SessionNotFoundError,
    StorageError,
)
from claude_agent_sdk.models import (
    SessionState,
    SessionMetadata,
    Message,
    CompactionState,
)
from claude_agent_sdk.serialization import (
    serialize_message,
    deserialize_message,
)


class RedisSessionStore:
    """
    Redis-based storage for distributed/stateless deployments.

    Schema:
    - session:{session_id}:messages -> List of JSON messages (RPUSH)
    - session:{session_id}:metadata -> Hash of metadata fields
    - session:{session_id}:created_at -> Timestamp string
    - sessions_by_dir:{working_dir} -> Set of session IDs (for filtering)

    All keys have TTL for automatic cleanup.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        default_ttl_days: int = 30
    ):
        """
        Initialize Redis store.

        Args:
            redis_url: Redis connection URL
            default_ttl_days: Default TTL for sessions in days
        """
        self.redis_url = redis_url
        self.redis: Optional[redis.Redis] = None
        self.default_ttl = timedelta(days=default_ttl_days)

    async def _ensure_connected(self):
        """Ensure Redis connection is established."""
        if self.redis is None:
            self.redis = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )

    def _message_key(self, session_id: str) -> str:
        """Get Redis key for message list."""
        return f"session:{session_id}:messages"

    def _metadata_key(self, session_id: str) -> str:
        """Get Redis key for metadata hash."""
        return f"session:{session_id}:metadata"

    def _created_at_key(self, session_id: str) -> str:
        """Get Redis key for created_at timestamp."""
        return f"session:{session_id}:created_at"

    def _directory_set_key(self, working_directory: str) -> str:
        """Get Redis key for directory-based session index."""
        return f"sessions_by_dir:{working_directory}"

    async def create_session(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> None:
        """
        Initialize a new session with metadata.

        Creates metadata hash and sets up indexes.
        """
        await self._ensure_connected()

        try:
            # Serialize metadata
            metadata_dict = metadata.to_dict()

            # Store metadata as hash
            metadata_key = self._metadata_key(session_id)
            await self.redis.hset(metadata_key, mapping=metadata_dict)

            # Set created_at timestamp
            created_at_key = self._created_at_key(session_id)
            await self.redis.set(created_at_key, datetime.now().isoformat())

            # Initialize empty message list (creates the key)
            message_key = self._message_key(session_id)
            await self.redis.lpush(message_key, "__init__")
            await self.redis.lpop(message_key)  # Remove init marker

            # Add to directory index
            dir_key = self._directory_set_key(metadata.working_directory)
            await self.redis.sadd(dir_key, session_id)

            # Set TTL on all keys
            ttl_seconds = int(self.default_ttl.total_seconds())
            await self.redis.expire(metadata_key, ttl_seconds)
            await self.redis.expire(created_at_key, ttl_seconds)
            await self.redis.expire(message_key, ttl_seconds)
            await self.redis.expire(dir_key, ttl_seconds)

        except Exception as e:
            raise StorageError(f"Failed to create session {session_id}: {e}") from e

    async def append_message(
        self,
        session_id: str,
        message: Message
    ) -> None:
        """
        Append a single message to session history.

        Uses RPUSH for atomic append to message list.
        """
        await self._ensure_connected()

        message_key = self._message_key(session_id)

        # Check if session exists
        exists = await self.redis.exists(message_key)
        if not exists:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        try:
            # Serialize message
            msg_data = serialize_message(message)
            msg_json = json.dumps(msg_data, ensure_ascii=False)

            # Atomic append to list
            await self.redis.rpush(message_key, msg_json)

            # Refresh TTL
            ttl_seconds = int(self.default_ttl.total_seconds())
            await self.redis.expire(message_key, ttl_seconds)

        except SessionNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to append message to {session_id}: {e}") from e

    async def load_session(
        self,
        session_id: str
    ) -> SessionState | None:
        """
        Load complete session state.

        Retrieves messages from list and metadata from hash.
        """
        await self._ensure_connected()

        message_key = self._message_key(session_id)
        metadata_key = self._metadata_key(session_id)

        # Check existence
        exists = await self.redis.exists(message_key)
        if not exists:
            return None

        try:
            # Load messages from list
            raw_messages = await self.redis.lrange(message_key, 0, -1)
            messages = [
                deserialize_message(json.loads(msg_json))
                for msg_json in raw_messages
            ]

            # Load metadata from hash
            metadata_dict = await self.redis.hgetall(metadata_key)
            if metadata_dict:
                # Convert string values back to correct types
                metadata_dict = self._restore_metadata_types(metadata_dict)
                metadata = SessionMetadata.from_dict(metadata_dict)
            else:
                metadata = SessionMetadata()

            # Load timestamps
            created_at_key = self._created_at_key(session_id)
            created_at_str = await self.redis.get(created_at_key)
            created_at = (
                datetime.fromisoformat(created_at_str)
                if created_at_str
                else datetime.now()
            )

            return SessionState(
                session_id=session_id,
                messages=messages,
                metadata=metadata,
                created_at=created_at,
                updated_at=datetime.now(),
            )

        except Exception as e:
            raise StorageError(f"Failed to load session {session_id}: {e}") from e

    def _restore_metadata_types(self, metadata_dict: dict) -> dict:
        """
        Restore proper types for metadata fields.

        Redis hashes store all values as strings, so we need to convert back.
        """
        # Convert numeric fields
        if "total_cost_usd" in metadata_dict:
            metadata_dict["total_cost_usd"] = float(metadata_dict["total_cost_usd"])
        if "num_turns" in metadata_dict:
            metadata_dict["num_turns"] = int(metadata_dict["num_turns"])

        # Convert JSON fields
        if "allowed_tools" in metadata_dict:
            metadata_dict["allowed_tools"] = json.loads(metadata_dict["allowed_tools"])
        if "disallowed_tools" in metadata_dict:
            metadata_dict["disallowed_tools"] = json.loads(metadata_dict["disallowed_tools"])
        if "usage" in metadata_dict:
            metadata_dict["usage"] = json.loads(metadata_dict["usage"])
        if "compaction_state" in metadata_dict and metadata_dict["compaction_state"] != "null":
            metadata_dict["compaction_state"] = json.loads(metadata_dict["compaction_state"])

        return metadata_dict

    async def session_exists(
        self,
        session_id: str
    ) -> bool:
        """
        Check if session exists without loading full state.
        """
        await self._ensure_connected()
        message_key = self._message_key(session_id)
        return await self.redis.exists(message_key) > 0

    async def update_metadata(
        self,
        session_id: str,
        metadata: SessionMetadata
    ) -> None:
        """
        Update session metadata.

        Updates the metadata hash in Redis.
        """
        await self._ensure_connected()

        metadata_key = self._metadata_key(session_id)

        # Check if session exists
        exists = await self.redis.exists(metadata_key)
        if not exists:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        try:
            # Serialize and update metadata
            metadata_dict = metadata.to_dict()
            await self.redis.hset(metadata_key, mapping=metadata_dict)

            # Refresh TTL
            ttl_seconds = int(self.default_ttl.total_seconds())
            await self.redis.expire(metadata_key, ttl_seconds)

        except Exception as e:
            raise StorageError(f"Failed to update metadata for {session_id}: {e}") from e

    async def close(self) -> None:
        """Clean up Redis connection."""
        if self.redis:
            await self.redis.close()
            self.redis = None
```

3. **Update `src/claude_agent_sdk/stores/__init__.py`** to export RedisSessionStore:
   ```python
   from claude_agent_sdk.stores.file_store import FileSessionStore
   from claude_agent_sdk.stores.redis_store import RedisSessionStore

   __all__ = ["FileSessionStore", "RedisSessionStore"]
   ```

4. **Install fakeredis for testing** - Add to `[dev]` dependencies: `fakeredis[lua]>=2.20.0`

5. **Create unit tests** in `tests/stores/test_redis_store.py`:

```python
"""Tests for RedisSessionStore."""

import pytest
from datetime import datetime
from fakeredis import aioredis as fakeredis
from claude_agent_sdk.stores.redis_store import RedisSessionStore
from claude_agent_sdk.models import SessionMetadata, Message
from claude_agent_sdk.protocols import SessionNotFoundError, StorageError


@pytest.fixture
async def redis_store():
    """Create a RedisSessionStore instance with fake Redis."""
    store = RedisSessionStore(redis_url="redis://fake")
    # Replace with fake Redis
    store.redis = await fakeredis.FakeRedis(decode_responses=True)
    yield store
    await store.close()


@pytest.mark.asyncio
async def test_create_session(redis_store):
    """Test creating a new session in Redis."""
    session_id = "test-redis-session"
    metadata = SessionMetadata(model="claude-sonnet-4-5-20250929")

    await redis_store.create_session(session_id, metadata)

    # Verify keys were created
    message_key = redis_store._message_key(session_id)
    metadata_key = redis_store._metadata_key(session_id)

    assert await redis_store.redis.exists(message_key)
    assert await redis_store.redis.exists(metadata_key)


@pytest.mark.asyncio
async def test_session_exists(redis_store):
    """Test checking if a session exists."""
    session_id = "exists-test"

    # Should not exist initially
    assert not await redis_store.session_exists(session_id)

    # Create session
    await redis_store.create_session(session_id, SessionMetadata())

    # Should exist now
    assert await redis_store.session_exists(session_id)


@pytest.mark.asyncio
async def test_append_message(redis_store):
    """Test appending a message to Redis."""
    session_id = "append-test"
    await redis_store.create_session(session_id, SessionMetadata())

    # Append message
    msg = Message(role="user", content="Hello Redis!", id="msg-1")
    await redis_store.append_message(session_id, msg)

    # Verify message was added to list
    message_key = redis_store._message_key(session_id)
    count = await redis_store.redis.llen(message_key)
    assert count == 1


@pytest.mark.asyncio
async def test_append_message_to_nonexistent_session(redis_store):
    """Test that appending to nonexistent session raises error."""
    msg = Message(role="user", content="Test")

    with pytest.raises(SessionNotFoundError):
        await redis_store.append_message("nonexistent", msg)


@pytest.mark.asyncio
async def test_load_session(redis_store):
    """Test loading a complete session from Redis."""
    session_id = "load-test"
    metadata = SessionMetadata(model="claude-opus-4", num_turns=2)

    # Create session
    await redis_store.create_session(session_id, metadata)

    # Append messages
    msg1 = Message(role="user", content="First", id="m1")
    msg2 = Message(role="assistant", content="Second", id="m2")
    await redis_store.append_message(session_id, msg1)
    await redis_store.append_message(session_id, msg2)

    # Load session
    session = await redis_store.load_session(session_id)

    assert session is not None
    assert session.session_id == session_id
    assert len(session.messages) == 2
    assert session.messages[0].content == "First"
    assert session.messages[1].content == "Second"
    assert session.metadata.model == "claude-opus-4"


@pytest.mark.asyncio
async def test_load_nonexistent_session(redis_store):
    """Test that loading nonexistent session returns None."""
    session = await redis_store.load_session("does-not-exist")
    assert session is None


@pytest.mark.asyncio
async def test_update_metadata(redis_store):
    """Test updating session metadata in Redis."""
    session_id = "metadata-test"
    metadata = SessionMetadata(num_turns=0, total_cost_usd=0.0)

    # Create session
    await redis_store.create_session(session_id, metadata)

    # Update metadata
    updated_metadata = SessionMetadata(num_turns=5, total_cost_usd=1.23)
    await redis_store.update_metadata(session_id, updated_metadata)

    # Load and verify
    session = await redis_store.load_session(session_id)
    assert session.metadata.num_turns == 5
    assert session.metadata.total_cost_usd == 1.23


@pytest.mark.asyncio
async def test_metadata_type_restoration(redis_store):
    """Test that metadata types are correctly restored from Redis strings."""
    session_id = "types-test"
    metadata = SessionMetadata(
        num_turns=10,
        total_cost_usd=5.67,
        allowed_tools=["tool1", "tool2"],
        usage={"input_tokens": 100, "output_tokens": 50}
    )

    await redis_store.create_session(session_id, metadata)

    # Load and verify types
    session = await redis_store.load_session(session_id)
    assert isinstance(session.metadata.num_turns, int)
    assert isinstance(session.metadata.total_cost_usd, float)
    assert isinstance(session.metadata.allowed_tools, list)
    assert isinstance(session.metadata.usage, dict)


@pytest.mark.asyncio
async def test_protocol_compliance(redis_store):
    """Test that RedisSessionStore satisfies SessionStore protocol."""
    from claude_agent_sdk.protocols import SessionStore

    assert isinstance(redis_store, SessionStore)
```

6. **Run the tests**:
   ```bash
   pytest tests/stores/test_redis_store.py -v --cov=src/claude_agent_sdk/stores/redis_store
   ```

7. **Update `src/claude_agent_sdk/__init__.py`** to export RedisSessionStore:
   ```python
   from claude_agent_sdk.stores import RedisSessionStore

   __all__ = [
       # ... existing exports ...
       "RedisSessionStore",
   ]
   ```

Ensure all tests pass and Redis store implements the basic SessionStore operations correctly.
</prompt>

---

#### Task 8: Implement RedisSessionStore Advanced Operations

Status: **Pending**

**Goal**: Complete the RedisSessionStore implementation with advanced operations including session listing, deletion, forking, compaction, and message range queries optimized for Redis data structures.

**Working Result**: A fully functional RedisSessionStore with all SessionStore protocol methods, leveraging Redis atomic operations (COPY, LRANGE, DEL) for efficient distributed storage.

**Validation**:
- [ ] `get_messages()` uses LRANGE for efficient message range retrieval
- [ ] `list_sessions()` scans keys with pattern matching, filters by directory using sets
- [ ] `delete_session()` deletes all related keys (messages, metadata, timestamps)
- [ ] `fork_session()` uses Redis COPY command for atomic session duplication
- [ ] `compact_session()` updates metadata with compaction state
- [ ] All operations handle Redis connection errors gracefully
- [ ] Unit tests cover all advanced operations with fakeredis
- [ ] `pytest tests/stores/test_redis_store.py -v` passes with >95% coverage

<prompt>
You are completing the RedisSessionStore implementation by adding advanced operations optimized for Redis's data structures and commands.

1. **Add the following methods to `src/claude_agent_sdk/stores/redis_store.py`**:

```python
    async def get_messages(
        self,
        session_id: str,
        from_turn: int = 0,
        to_turn: int | None = None
    ) -> list[Message]:
        """
        Retrieve message history with optional range.

        Uses Redis LRANGE for efficient range queries.
        """
        await self._ensure_connected()

        message_key = self._message_key(session_id)

        exists = await self.redis.exists(message_key)
        if not exists:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        try:
            # Redis LRANGE with negative indices for end of list
            if to_turn is None:
                raw_messages = await self.redis.lrange(message_key, from_turn, -1)
            else:
                raw_messages = await self.redis.lrange(message_key, from_turn, to_turn - 1)

            messages = [
                deserialize_message(json.loads(msg_json))
                for msg_json in raw_messages
            ]
            return messages

        except Exception as e:
            raise StorageError(f"Failed to get messages from {session_id}: {e}") from e

    async def list_sessions(
        self,
        working_directory: str | None = None,
        limit: int = 100
    ) -> list[str]:
        """
        List session IDs, optionally filtered by working directory.

        Uses Redis key scanning and directory sets.
        """
        await self._ensure_connected()

        try:
            if working_directory is not None:
                # Use directory index set
                dir_key = self._directory_set_key(working_directory)
                session_ids = await self.redis.smembers(dir_key)
                # Convert set to list and limit
                return list(session_ids)[:limit]
            else:
                # Scan all session message keys
                session_ids = []
                cursor = 0

                while True:
                    cursor, keys = await self.redis.scan(
                        cursor,
                        match="session:*:messages",
                        count=100
                    )

                    # Extract session IDs from keys
                    for key in keys:
                        # Parse "session:{session_id}:messages"
                        parts = key.split(":")
                        if len(parts) >= 3:
                            session_id = parts[1]
                            session_ids.append(session_id)

                    if cursor == 0:
                        break

                    if len(session_ids) >= limit:
                        break

                return session_ids[:limit]

        except Exception as e:
            raise StorageError(f"Failed to list sessions: {e}") from e

    async def delete_session(
        self,
        session_id: str
    ) -> None:
        """
        Permanently delete a session.

        Removes all related Redis keys.
        """
        await self._ensure_connected()

        message_key = self._message_key(session_id)

        exists = await self.redis.exists(message_key)
        if not exists:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        try:
            # Get working directory before deleting
            metadata_key = self._metadata_key(session_id)
            metadata_dict = await self.redis.hgetall(metadata_key)

            # Delete all session keys
            created_at_key = self._created_at_key(session_id)
            await self.redis.delete(message_key, metadata_key, created_at_key)

            # Remove from directory index if we have metadata
            if metadata_dict and "working_directory" in metadata_dict:
                dir_key = self._directory_set_key(metadata_dict["working_directory"])
                await self.redis.srem(dir_key, session_id)

        except Exception as e:
            raise StorageError(f"Failed to delete session {session_id}: {e}") from e

    async def fork_session(
        self,
        source_session_id: str,
        new_session_id: str
    ) -> None:
        """
        Create a copy of a session with a new ID.

        Uses Redis COPY command for atomic duplication.
        """
        await self._ensure_connected()

        # Verify source exists
        exists = await self.session_exists(source_session_id)
        if not exists:
            raise SessionNotFoundError(f"Source session {source_session_id} does not exist")

        try:
            # Copy all keys with new session ID
            source_message_key = self._message_key(source_session_id)
            source_metadata_key = self._metadata_key(source_session_id)
            source_created_at_key = self._created_at_key(source_session_id)

            dest_message_key = self._message_key(new_session_id)
            dest_metadata_key = self._metadata_key(new_session_id)
            dest_created_at_key = self._created_at_key(new_session_id)

            # Copy message list
            # Note: COPY command requires Redis 6.2+, fallback to manual copy
            try:
                await self.redis.copy(source_message_key, dest_message_key)
            except Exception:
                # Manual copy for older Redis versions
                messages = await self.redis.lrange(source_message_key, 0, -1)
                if messages:
                    await self.redis.rpush(dest_message_key, *messages)

            # Copy metadata hash
            try:
                await self.redis.copy(source_metadata_key, dest_metadata_key)
            except Exception:
                # Manual copy
                metadata = await self.redis.hgetall(source_metadata_key)
                if metadata:
                    await self.redis.hset(dest_metadata_key, mapping=metadata)

            # Set new created_at
            await self.redis.set(dest_created_at_key, datetime.now().isoformat())

            # Add to directory index
            metadata_dict = await self.redis.hgetall(dest_metadata_key)
            if metadata_dict and "working_directory" in metadata_dict:
                dir_key = self._directory_set_key(metadata_dict["working_directory"])
                await self.redis.sadd(dir_key, new_session_id)

            # Set TTL on new keys
            ttl_seconds = int(self.default_ttl.total_seconds())
            await self.redis.expire(dest_message_key, ttl_seconds)
            await self.redis.expire(dest_metadata_key, ttl_seconds)
            await self.redis.expire(dest_created_at_key, ttl_seconds)

        except SessionNotFoundError:
            raise
        except Exception as e:
            raise StorageError(f"Failed to fork session {source_session_id}: {e}") from e

    async def compact_session(
        self,
        session_id: str,
        compaction_state: CompactionState
    ) -> None:
        """
        Apply context compaction to session.

        Updates metadata with compaction state.
        """
        # Load current session
        session = await self.load_session(session_id)
        if session is None:
            raise SessionNotFoundError(f"Session {session_id} does not exist")

        # Update metadata with compaction state
        session.metadata.compaction_state = compaction_state

        # Update metadata in Redis
        await self.update_metadata(session_id, session.metadata)
```

2. **Add comprehensive tests for advanced operations** in `tests/stores/test_redis_store.py`:

```python
@pytest.mark.asyncio
async def test_get_messages_full_range(redis_store):
    """Test getting all messages from a session."""
    session_id = "messages-test"
    await redis_store.create_session(session_id, SessionMetadata())

    # Add several messages
    for i in range(5):
        await redis_store.append_message(
            session_id,
            Message(role="user", content=f"Message {i}", id=f"m{i}")
        )

    messages = await redis_store.get_messages(session_id)
    assert len(messages) == 5
    assert messages[0].content == "Message 0"
    assert messages[4].content == "Message 4"


@pytest.mark.asyncio
async def test_get_messages_with_range(redis_store):
    """Test getting a subset of messages using range."""
    session_id = "range-test"
    await redis_store.create_session(session_id, SessionMetadata())

    for i in range(10):
        await redis_store.append_message(
            session_id,
            Message(role="user", content=f"Msg {i}", id=f"m{i}")
        )

    # Get messages 3-7
    messages = await redis_store.get_messages(session_id, from_turn=3, to_turn=7)
    assert len(messages) == 4
    assert messages[0].content == "Msg 3"
    assert messages[3].content == "Msg 6"


@pytest.mark.asyncio
async def test_list_sessions(redis_store):
    """Test listing all sessions."""
    # Create multiple sessions
    for i in range(3):
        await redis_store.create_session(f"session-{i}", SessionMetadata())

    sessions = await redis_store.list_sessions()
    assert len(sessions) == 3


@pytest.mark.asyncio
async def test_list_sessions_by_directory(redis_store):
    """Test filtering sessions by working directory."""
    metadata1 = SessionMetadata(working_directory="/tmp/project1")
    metadata2 = SessionMetadata(working_directory="/tmp/project2")

    await redis_store.create_session("session-1", metadata1)
    await redis_store.create_session("session-2", metadata2)
    await redis_store.create_session("session-3", metadata1)

    # List sessions in project1
    sessions = await redis_store.list_sessions(working_directory="/tmp/project1")
    assert len(sessions) == 2
    assert "session-1" in sessions
    assert "session-3" in sessions


@pytest.mark.asyncio
async def test_delete_session(redis_store):
    """Test deleting a session from Redis."""
    session_id = "delete-test"
    await redis_store.create_session(session_id, SessionMetadata())

    # Verify exists
    assert await redis_store.session_exists(session_id)

    # Delete
    await redis_store.delete_session(session_id)

    # Verify gone
    assert not await redis_store.session_exists(session_id)


@pytest.mark.asyncio
async def test_delete_removes_all_keys(redis_store):
    """Test that deletion removes all related keys."""
    session_id = "delete-keys-test"
    await redis_store.create_session(session_id, SessionMetadata())
    await redis_store.append_message(
        session_id,
        Message(role="user", content="Test", id="m1")
    )

    # Get key names
    message_key = redis_store._message_key(session_id)
    metadata_key = redis_store._metadata_key(session_id)
    created_at_key = redis_store._created_at_key(session_id)

    # Delete
    await redis_store.delete_session(session_id)

    # Verify all keys gone
    assert not await redis_store.redis.exists(message_key)
    assert not await redis_store.redis.exists(metadata_key)
    assert not await redis_store.redis.exists(created_at_key)


@pytest.mark.asyncio
async def test_fork_session(redis_store):
    """Test forking a session in Redis."""
    source_id = "source-session"
    fork_id = "forked-session"

    # Create source with messages
    metadata = SessionMetadata(model="claude-opus-4")
    await redis_store.create_session(source_id, metadata)
    await redis_store.append_message(
        source_id,
        Message(role="user", content="Original", id="m1")
    )

    # Fork
    await redis_store.fork_session(source_id, fork_id)

    # Verify fork exists and has same content
    fork_session = await redis_store.load_session(fork_id)
    assert fork_session is not None
    assert len(fork_session.messages) == 1
    assert fork_session.messages[0].content == "Original"
    assert fork_session.metadata.model == "claude-opus-4"

    # Verify source unchanged
    source_session = await redis_store.load_session(source_id)
    assert source_session is not None


@pytest.mark.asyncio
async def test_fork_nonexistent_session(redis_store):
    """Test that forking nonexistent session raises error."""
    with pytest.raises(SessionNotFoundError):
        await redis_store.fork_session("nonexistent", "new-fork")


@pytest.mark.asyncio
async def test_compact_session(redis_store):
    """Test applying context compaction."""
    from claude_agent_sdk.models import CompactionState

    session_id = "compact-test"
    await redis_store.create_session(session_id, SessionMetadata())

    for i in range(5):
        await redis_store.append_message(
            session_id,
            Message(role="user", content=f"Msg {i}", id=f"m{i}")
        )

    # Apply compaction
    compaction = CompactionState(
        last_compaction_turn=3,
        summary="Summary of early messages",
        original_message_ids=["m0", "m1", "m2"]
    )
    await redis_store.compact_session(session_id, compaction)

    # Verify compaction saved
    session = await redis_store.load_session(session_id)
    assert session.metadata.compaction_state is not None
    assert session.metadata.compaction_state.last_compaction_turn == 3


@pytest.mark.asyncio
async def test_connection_management(redis_store):
    """Test that Redis connection is properly managed."""
    # Initially not connected
    assert redis_store.redis is not None  # Already set in fixture

    # Operations should work
    await redis_store.create_session("test", SessionMetadata())

    # Close should clean up
    await redis_store.close()
    assert redis_store.redis is None
```

3. **Run all Redis store tests**:
   ```bash
   pytest tests/stores/test_redis_store.py -v --cov=src/claude_agent_sdk/stores/redis_store --cov-report=term-missing
   ```

4. **Verify protocol compliance**:
   ```bash
   mypy src/claude_agent_sdk/stores/redis_store.py --strict
   ```

Ensure all tests pass with high coverage and RedisSessionStore fully implements the SessionStore protocol with Redis-optimized operations.
</prompt>

---

### 🔄 **Iteration 4: Stateless Agent Executor**

This iteration implements the AgentExecutor, the core stateless execution engine that orchestrates agent behavior. The executor loads state from SessionStore, executes agent turns, and persists results back to storage without maintaining any internal state.

---

#### Task 9: Implement AgentExecutor Core Structure

Status: **Pending**

**Goal**: Create the foundational AgentExecutor class that loads session state from SessionStore, executes a single agent turn by calling Claude API, and persists messages back to storage in a completely stateless manner.

**Working Result**: A working **AgentExecutor** class in `src/claude_agent_sdk/executor.py` that can execute a simple single-turn conversation (user message → Claude API → assistant response) with state loaded from and saved to any SessionStore implementation.

**Validation**:
- [ ] `AgentExecutor` class exists in `src/claude_agent_sdk/executor.py`
- [ ] Constructor accepts `SessionStore`, `Anthropic` client, and `ClaudeAgentOptions`
- [ ] `execute_turn()` method loads session, sends user message, calls Claude API
- [ ] All messages (user and assistant) are persisted to SessionStore
- [ ] Metadata (usage, costs, turn count) is updated after each turn
- [ ] Executor has no instance state - all state comes from SessionStore
- [ ] Unit tests use mock SessionStore and mock Anthropic client
- [ ] `pytest tests/test_executor.py::test_basic_turn -v` passes

<prompt>
You are implementing the AgentExecutor, the stateless core of the Claude Agent SDK. This class orchestrates agent execution without maintaining any internal state.

1. **Create `src/claude_agent_sdk/executor.py`** with the following implementation:

```python
"""Stateless agent execution engine."""

import asyncio
from typing import AsyncIterator, Optional
from dataclasses import dataclass
from anthropic import Anthropic, AsyncAnthropic
from anthropic.types import Message as AnthropicMessage, MessageStreamEvent

from claude_agent_sdk.protocols import SessionStore
from claude_agent_sdk.models import (
    SessionState,
    SessionMetadata,
    Message,
    PermissionMode,
)


@dataclass
class ClaudeAgentOptions:
    """Configuration options for agent execution."""

    model: str = "claude-sonnet-4-5-20250929"
    """Claude model to use"""

    session_store: Optional[SessionStore] = None
    """Storage backend (defaults to FileSessionStore)"""

    working_directory: str = "."
    """Working directory for file operations"""

    permission_mode: PermissionMode = PermissionMode.ASK
    """How to handle tool execution"""

    max_turns: int = 100
    """Maximum number of conversation turns"""

    resume: Optional[str] = None
    """Session ID to resume"""

    anthropic_api_key: Optional[str] = None
    """Anthropic API key (defaults to ANTHROPIC_API_KEY env var)"""


class AgentExecutor:
    """
    Stateless agent execution engine.

    All state is loaded from SessionStore at the start of each operation
    and persisted back after each message. The executor maintains NO
    internal state between calls.
    """

    def __init__(
        self,
        session_store: SessionStore,
        anthropic_client: AsyncAnthropic,
        options: ClaudeAgentOptions
    ):
        """
        Initialize executor.

        Args:
            session_store: Storage backend for session state
            anthropic_client: Anthropic API client
            options: Agent configuration options
        """
        self.store = session_store
        self.client = anthropic_client
        self.options = options

    async def execute_turn(
        self,
        session_id: str,
        user_message: str
    ) -> AsyncIterator[Message]:
        """
        Execute a single agent turn.

        This is the main entry point for agent execution. It:
        1. Loads session state from store
        2. Appends user message
        3. Calls Claude API and streams response
        4. Persists assistant message
        5. Updates metadata

        Args:
            session_id: Session ID to execute turn in
            user_message: User's message text

        Yields:
            Messages as they are created (user message, assistant message, etc.)
        """
        # Load or create session
        session = await self.store.load_session(session_id)
        if session is None:
            session = await self._initialize_session(session_id)

        # Check turn limit
        if session.metadata.num_turns >= self.options.max_turns:
            # TODO: Yield a result message indicating max turns reached
            return

        # Create and persist user message
        user_msg = Message(
            role="user",
            content=user_message,
            id=f"msg-{session.metadata.num_turns}-user"
        )
        await self.store.append_message(session_id, user_msg)
        yield user_msg

        # Build API request messages (convert from our Message type to Anthropic format)
        api_messages = self._build_api_messages(session.messages + [user_msg])

        # Call Claude API (streaming)
        stream = await self.client.messages.create(
            model=self.options.model,
            max_tokens=4096,
            messages=api_messages,
            stream=True
        )

        # Process response stream
        assistant_content = []
        async for event in stream:
            if event.type == "content_block_start":
                if event.content_block.type == "text":
                    assistant_content.append({"type": "text", "text": ""})
            elif event.type == "content_block_delta":
                if event.delta.type == "text_delta":
                    # Accumulate text
                    if assistant_content:
                        assistant_content[-1]["text"] += event.delta.text
            elif event.type == "message_stop":
                # Message complete
                break

        # Create assistant message
        assistant_msg = Message(
            role="assistant",
            content=assistant_content if assistant_content else "...",
            id=f"msg-{session.metadata.num_turns}-assistant"
        )

        # Persist assistant message
        await self.store.append_message(session_id, assistant_msg)
        yield assistant_msg

        # Update metadata
        session.metadata.num_turns += 1
        # TODO: Update costs and usage from API response
        await self.store.update_metadata(session_id, session.metadata)

    async def _initialize_session(self, session_id: str) -> SessionState:
        """
        Initialize a new session.

        Creates session with default metadata in the store.
        """
        metadata = SessionMetadata(
            model=self.options.model,
            working_directory=self.options.working_directory,
            permission_mode=self.options.permission_mode,
        )

        await self.store.create_session(session_id, metadata)

        # Load and return the newly created session
        session = await self.store.load_session(session_id)
        if session is None:
            raise RuntimeError(f"Failed to create session {session_id}")

        return session

    def _build_api_messages(self, messages: list[Message]) -> list[dict]:
        """
        Convert our Message format to Anthropic API format.

        Args:
            messages: List of our Message objects

        Returns:
            List of dicts suitable for Anthropic API
        """
        api_messages = []
        for msg in messages:
            api_msg = {
                "role": msg.role,
                "content": msg.content
            }
            api_messages.append(api_msg)

        return api_messages
```

2. **Create unit tests** in `tests/test_executor.py`:

```python
"""Tests for AgentExecutor."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from claude_agent_sdk.executor import AgentExecutor, ClaudeAgentOptions
from claude_agent_sdk.models import SessionState, SessionMetadata, Message
from claude_agent_sdk.protocols import SessionStore


class MockSessionStore:
    """Mock SessionStore for testing."""

    def __init__(self):
        self.sessions = {}
        self.messages = {}

    async def create_session(self, session_id: str, metadata: SessionMetadata):
        self.sessions[session_id] = metadata
        self.messages[session_id] = []

    async def load_session(self, session_id: str) -> SessionState | None:
        if session_id not in self.sessions:
            return None

        return SessionState(
            session_id=session_id,
            messages=self.messages.get(session_id, []),
            metadata=self.sessions[session_id],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

    async def append_message(self, session_id: str, message: Message):
        if session_id not in self.messages:
            self.messages[session_id] = []
        self.messages[session_id].append(message)

    async def update_metadata(self, session_id: str, metadata: SessionMetadata):
        self.sessions[session_id] = metadata

    async def session_exists(self, session_id: str) -> bool:
        return session_id in self.sessions

    async def get_messages(self, session_id: str, from_turn: int = 0, to_turn: int | None = None):
        return self.messages.get(session_id, [])[from_turn:to_turn]

    async def delete_session(self, session_id: str):
        self.sessions.pop(session_id, None)
        self.messages.pop(session_id, None)

    async def fork_session(self, source_session_id: str, new_session_id: str):
        pass

    async def compact_session(self, session_id: str, compaction_state):
        pass

    async def list_sessions(self, working_directory: str | None = None, limit: int = 100):
        return list(self.sessions.keys())

    async def close(self):
        pass


class MockAnthropicMessage:
    """Mock streaming event from Anthropic API."""

    def __init__(self, event_type: str, **kwargs):
        self.type = event_type
        for key, value in kwargs.items():
            setattr(self, key, value)


class MockTextBlock:
    """Mock text content block."""
    def __init__(self):
        self.type = "text"


class MockTextDelta:
    """Mock text delta."""
    def __init__(self, text: str):
        self.type = "text_delta"
        self.text = text


@pytest.fixture
def mock_store():
    """Create a mock session store."""
    return MockSessionStore()


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client."""
    client = AsyncMock()
    return client


@pytest.fixture
def executor(mock_store, mock_anthropic_client):
    """Create an AgentExecutor for testing."""
    options = ClaudeAgentOptions(
        model="claude-sonnet-4-5-20250929",
        max_turns=10
    )
    return AgentExecutor(
        session_store=mock_store,
        anthropic_client=mock_anthropic_client,
        options=options
    )


@pytest.mark.asyncio
async def test_initialize_new_session(executor, mock_store):
    """Test that executor initializes a new session if it doesn't exist."""
    session_id = "new-session"

    # Session shouldn't exist yet
    assert not await mock_store.session_exists(session_id)

    # Mock the Anthropic API response
    mock_stream = [
        MockAnthropicMessage("content_block_start", content_block=MockTextBlock()),
        MockAnthropicMessage("content_block_delta", delta=MockTextDelta("Hello")),
        MockAnthropicMessage("content_block_delta", delta=MockTextDelta(" there!")),
        MockAnthropicMessage("message_stop"),
    ]

    executor.client.messages.create = AsyncMock()
    executor.client.messages.create.return_value = async_generator_from_list(mock_stream)

    # Execute turn
    messages = []
    async for msg in executor.execute_turn(session_id, "Hi"):
        messages.append(msg)

    # Verify session was created
    assert await mock_store.session_exists(session_id)

    # Verify messages were persisted
    assert len(messages) == 2  # user + assistant
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_execute_basic_turn(executor, mock_store):
    """Test executing a basic single turn."""
    session_id = "test-session"

    # Mock Anthropic API
    mock_stream = [
        MockAnthropicMessage("content_block_start", content_block=MockTextBlock()),
        MockAnthropicMessage("content_block_delta", delta=MockTextDelta("Response")),
        MockAnthropicMessage("message_stop"),
    ]

    executor.client.messages.create = AsyncMock()
    executor.client.messages.create.return_value = async_generator_from_list(mock_stream)

    # Execute turn
    messages = []
    async for msg in executor.execute_turn(session_id, "Test message"):
        messages.append(msg)

    # Verify messages
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[0].content == "Test message"
    assert messages[1].role == "assistant"
    assert messages[1].content[0]["text"] == "Response"


@pytest.mark.asyncio
async def test_messages_persisted_to_store(executor, mock_store):
    """Test that all messages are persisted to the session store."""
    session_id = "persist-test"

    # Mock API
    mock_stream = [
        MockAnthropicMessage("content_block_start", content_block=MockTextBlock()),
        MockAnthropicMessage("content_block_delta", delta=MockTextDelta("OK")),
        MockAnthropicMessage("message_stop"),
    ]
    executor.client.messages.create = AsyncMock()
    executor.client.messages.create.return_value = async_generator_from_list(mock_stream)

    # Execute turn
    async for _ in executor.execute_turn(session_id, "Hello"):
        pass

    # Load session and check messages
    session = await mock_store.load_session(session_id)
    assert len(session.messages) == 2
    assert session.messages[0].role == "user"
    assert session.messages[1].role == "assistant"


@pytest.mark.asyncio
async def test_metadata_updated_after_turn(executor, mock_store):
    """Test that metadata is updated after each turn."""
    session_id = "metadata-test"

    # Mock API
    mock_stream = [
        MockAnthropicMessage("content_block_start", content_block=MockTextBlock()),
        MockAnthropicMessage("content_block_delta", delta=MockTextDelta("Reply")),
        MockAnthropicMessage("message_stop"),
    ]
    executor.client.messages.create = AsyncMock()
    executor.client.messages.create.return_value = async_generator_from_list(mock_stream)

    # Execute turn
    async for _ in executor.execute_turn(session_id, "Message"):
        pass

    # Check metadata
    session = await mock_store.load_session(session_id)
    assert session.metadata.num_turns == 1


@pytest.mark.asyncio
async def test_resume_existing_session(executor, mock_store):
    """Test resuming an existing session."""
    session_id = "resume-test"

    # Create initial session
    metadata = SessionMetadata(num_turns=0)
    await mock_store.create_session(session_id, metadata)

    # Add initial message
    first_msg = Message(role="user", content="First", id="m1")
    await mock_store.append_message(session_id, first_msg)

    # Mock API for second turn
    mock_stream = [
        MockAnthropicMessage("content_block_start", content_block=MockTextBlock()),
        MockAnthropicMessage("content_block_delta", delta=MockTextDelta("Second response")),
        MockAnthropicMessage("message_stop"),
    ]
    executor.client.messages.create = AsyncMock()
    executor.client.messages.create.return_value = async_generator_from_list(mock_stream)

    # Execute second turn
    async for _ in executor.execute_turn(session_id, "Second message"):
        pass

    # Verify session has both conversations
    session = await mock_store.load_session(session_id)
    # Should have: first user, second user, second assistant
    assert len(session.messages) >= 2


@pytest.mark.asyncio
async def test_max_turns_limit(executor, mock_store):
    """Test that execution stops at max turns limit."""
    session_id = "max-turns-test"

    # Create session at max turns
    metadata = SessionMetadata(num_turns=10)  # executor max_turns is 10
    await mock_store.create_session(session_id, metadata)

    # Try to execute another turn
    messages = []
    async for msg in executor.execute_turn(session_id, "This should not execute"):
        messages.append(msg)

    # Should not execute (no messages)
    assert len(messages) == 0


def async_generator_from_list(items):
    """Helper to create async generator from list."""
    async def _gen():
        for item in items:
            yield item
    return _gen()
```

3. **Run the executor tests**:
   ```bash
   pytest tests/test_executor.py -v --cov=src/claude_agent_sdk/executor
   ```

4. **Update `src/claude_agent_sdk/__init__.py`** to export executor components:
   ```python
   from claude_agent_sdk.executor import AgentExecutor, ClaudeAgentOptions

   __all__ = [
       # ... existing exports ...
       "AgentExecutor",
       "ClaudeAgentOptions",
   ]
   ```

Ensure all tests pass and the executor correctly executes single turns in a stateless manner.
</prompt>

---

### 🔄 **Iteration 5: Public API Integration**

This final iteration creates the user-facing API that maintains 100% backward compatibility with existing Claude SDK patterns while leveraging the new stateless architecture. Users can choose their storage backend or use the default file-based storage.

---

#### Task 10: Implement Public API Functions

Status: **Pending**

**Goal**: Create the public API entry points (`query()` function and `ClaudeSDKClient` class) that provide a simple, intuitive interface for agent execution while supporting pluggable storage backends.

**Working Result**: A complete public API in `src/claude_agent_sdk/api.py` with the `query()` function for single queries and `ClaudeSDKClient` class for multi-turn conversations, both using AgentExecutor and SessionStore under the hood.

**Validation**:
- [ ] `query()` async function exists for single stateless queries
- [ ] `ClaudeSDKClient` class exists for multi-turn conversations
- [ ] `query()` accepts options with custom `session_store` parameter
- [ ] Default storage backend is `FileSessionStore` when none specified
- [ ] `ClaudeSDKClient` maintains session across multiple `query()` calls
- [ ] Both API patterns support `resume` parameter for session continuation
- [ ] Integration tests demonstrate file store and Redis store usage
- [ ] `pytest tests/test_api.py -v` passes with full integration tests

<prompt>
You are implementing the public API for the Stateless Claude Agent SDK. This API provides backward-compatible entry points while supporting pluggable storage.

1. **Create `src/claude_agent_sdk/api.py`** with the following implementation:

```python
"""Public API for Claude Agent SDK."""

import uuid
from typing import AsyncIterator, Optional
from anthropic import AsyncAnthropic

from claude_agent_sdk.executor import AgentExecutor, ClaudeAgentOptions
from claude_agent_sdk.models import Message
from claude_agent_sdk.stores.file_store import FileSessionStore


async def query(
    prompt: str,
    options: Optional[ClaudeAgentOptions] = None
) -> AsyncIterator[Message]:
    """
    Execute a single agent query.

    This is the simplest entry point for one-off agent queries.
    For multi-turn conversations, use ClaudeSDKClient instead.

    Args:
        prompt: User's message/prompt
        options: Optional configuration (model, storage backend, etc.)

    Yields:
        Messages as they are created during execution

    Examples:
        >>> async for message in query("Hello, Claude!"):
        ...     print(message.content)

        >>> # Use Redis storage
        >>> from claude_agent_sdk.stores import RedisSessionStore
        >>> options = ClaudeAgentOptions(
        ...     session_store=RedisSessionStore("redis://localhost:6379")
        ... )
        >>> async for message in query("Hello", options=options):
        ...     print(message.content)
    """
    if options is None:
        options = ClaudeAgentOptions()

    # Set up storage backend
    if options.session_store is None:
        options.session_store = FileSessionStore()

    # Set up Anthropic client
    anthropic_client = AsyncAnthropic(api_key=options.anthropic_api_key)

    # Create executor
    executor = AgentExecutor(
        session_store=options.session_store,
        anthropic_client=anthropic_client,
        options=options
    )

    # Generate session ID or use resume ID
    session_id = options.resume or str(uuid.uuid4())

    # Execute query
    async for message in executor.execute_turn(session_id, prompt):
        yield message


class ClaudeSDKClient:
    """
    Client for multi-turn agent conversations.

    Maintains a single session across multiple query calls,
    enabling conversational context to be preserved.

    Examples:
        >>> client = ClaudeSDKClient()
        >>> await client.query("What's 2+2?")
        >>> await client.query("And what's that times 3?")
        >>> await client.close()

        >>> # Use Redis storage
        >>> options = ClaudeAgentOptions(
        ...     session_store=RedisSessionStore("redis://localhost")
        ... )
        >>> client = ClaudeSDKClient(options=options)

        >>> # Use as async context manager
        >>> async with ClaudeSDKClient() as client:
        ...     await client.query("Hello")
    """

    def __init__(self, options: Optional[ClaudeAgentOptions] = None):
        """
        Initialize client.

        Args:
            options: Optional configuration
        """
        self.options = options or ClaudeAgentOptions()

        # Set up storage backend
        if self.options.session_store is None:
            self.options.session_store = FileSessionStore()

        # Set up Anthropic client
        self.anthropic_client = AsyncAnthropic(
            api_key=self.options.anthropic_api_key
        )

        # Create executor
        self.executor = AgentExecutor(
            session_store=self.options.session_store,
            anthropic_client=self.anthropic_client,
            options=self.options
        )

        # Session ID for this client
        self.session_id = self.options.resume or str(uuid.uuid4())

    async def query(self, prompt: str) -> AsyncIterator[Message]:
        """
        Execute a query in this client's session.

        Args:
            prompt: User's message

        Yields:
            Messages as they are created
        """
        async for message in self.executor.execute_turn(self.session_id, prompt):
            yield message

    async def get_session_id(self) -> str:
        """
        Get the current session ID.

        Returns:
            Session ID being used by this client
        """
        return self.session_id

    async def close(self):
        """
        Clean up resources.

        Closes the storage backend connection.
        """
        if hasattr(self.options.session_store, 'close'):
            await self.options.session_store.close()

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
```

2. **Update `src/claude_agent_sdk/__init__.py`** to export the public API:

```python
"""Claude Agent SDK - Stateless agent execution with pluggable storage."""

__version__ = "0.1.0"

# Core models
from claude_agent_sdk.models import (
    SessionState,
    SessionMetadata,
    CompactionState,
    Message,
    PermissionMode,
)

# Protocols
from claude_agent_sdk.protocols import (
    SessionStore,
    SessionNotFoundError,
    StorageError,
    ConcurrencyError,
)

# Storage backends
from claude_agent_sdk.stores import FileSessionStore, RedisSessionStore

# Executor
from claude_agent_sdk.executor import AgentExecutor, ClaudeAgentOptions

# Public API
from claude_agent_sdk.api import query, ClaudeSDKClient

__all__ = [
    # Version
    "__version__",
    # Models
    "SessionState",
    "SessionMetadata",
    "CompactionState",
    "Message",
    "PermissionMode",
    # Protocols
    "SessionStore",
    "SessionNotFoundError",
    "StorageError",
    "ConcurrencyError",
    # Stores
    "FileSessionStore",
    "RedisSessionStore",
    # Executor
    "AgentExecutor",
    "ClaudeAgentOptions",
    # Public API
    "query",
    "ClaudeSDKClient",
]
```

3. **Create integration tests** in `tests/test_api.py`:

```python
"""Integration tests for public API."""

import pytest
from unittest.mock import AsyncMock, patch
from claude_agent_sdk.api import query, ClaudeSDKClient
from claude_agent_sdk.executor import ClaudeAgentOptions
from claude_agent_sdk.stores.file_store import FileSessionStore


# Helper to create mock Anthropic stream
def mock_anthropic_stream(text: str):
    """Create a mock Anthropic API stream."""
    class MockContentBlock:
        type = "text"

    class MockDelta:
        type = "text_delta"
        def __init__(self, text):
            self.text = text

    class MockEvent:
        def __init__(self, event_type, **kwargs):
            self.type = event_type
            for k, v in kwargs.items():
                setattr(self, k, v)

    async def stream():
        yield MockEvent("content_block_start", content_block=MockContentBlock())
        yield MockEvent("content_block_delta", delta=MockDelta(text))
        yield MockEvent("message_stop")

    return stream()


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_query_basic(mock_anthropic, tmp_path):
    """Test basic query() function."""
    # Setup mock
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.create.return_value = mock_anthropic_stream("Hello!")
    mock_anthropic.return_value = mock_client

    # Use temporary file store
    options = ClaudeAgentOptions(
        session_store=FileSessionStore(base_path=tmp_path)
    )

    # Execute query
    messages = []
    async for message in query("Hi", options=options):
        messages.append(message)

    # Verify
    assert len(messages) == 2  # user + assistant
    assert messages[0].role == "user"
    assert messages[0].content == "Hi"
    assert messages[1].role == "assistant"


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_query_with_file_store(mock_anthropic, tmp_path):
    """Test that query() uses FileSessionStore by default."""
    # Setup mock
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.create.return_value = mock_anthropic_stream("Response")
    mock_anthropic.return_value = mock_client

    # Execute query with file store
    options = ClaudeAgentOptions(
        session_store=FileSessionStore(base_path=tmp_path)
    )

    async for _ in query("Test", options=options):
        pass

    # Verify session was saved to file system
    projects_dir = tmp_path / "projects"
    assert projects_dir.exists()


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_query_resume_session(mock_anthropic, tmp_path):
    """Test resuming a session with query()."""
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_anthropic.return_value = mock_client

    store = FileSessionStore(base_path=tmp_path)
    session_id = "test-resume-session"

    # First query
    mock_client.messages.create.return_value = mock_anthropic_stream("First response")
    options1 = ClaudeAgentOptions(session_store=store, resume=session_id)

    async for _ in query("First message", options=options1):
        pass

    # Second query (resume)
    mock_client.messages.create.return_value = mock_anthropic_stream("Second response")
    options2 = ClaudeAgentOptions(session_store=store, resume=session_id)

    async for _ in query("Second message", options=options2):
        pass

    # Verify session has multiple messages
    session = await store.load_session(session_id)
    assert len(session.messages) >= 3  # First user/assistant + second user


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_client_multi_turn(mock_anthropic, tmp_path):
    """Test ClaudeSDKClient for multi-turn conversation."""
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_anthropic.return_value = mock_client

    options = ClaudeAgentOptions(
        session_store=FileSessionStore(base_path=tmp_path)
    )

    client = ClaudeSDKClient(options=options)

    # First turn
    mock_client.messages.create.return_value = mock_anthropic_stream("Response 1")
    messages1 = []
    async for msg in client.query("Message 1"):
        messages1.append(msg)

    # Second turn (same session)
    mock_client.messages.create.return_value = mock_anthropic_stream("Response 2")
    messages2 = []
    async for msg in client.query("Message 2"):
        messages2.append(msg)

    await client.close()

    # Verify both turns executed
    assert len(messages1) == 2
    assert len(messages2) == 2

    # Verify session preserved across calls
    session_id = await client.get_session_id()
    session = await options.session_store.load_session(session_id)
    assert len(session.messages) >= 4  # 2 turns x 2 messages each


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_client_context_manager(mock_anthropic, tmp_path):
    """Test ClaudeSDKClient as async context manager."""
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.create.return_value = mock_anthropic_stream("Response")
    mock_anthropic.return_value = mock_client

    options = ClaudeAgentOptions(
        session_store=FileSessionStore(base_path=tmp_path)
    )

    async with ClaudeSDKClient(options=options) as client:
        async for _ in client.query("Test"):
            pass

    # Context manager should have called close()
    # File store doesn't track this, but no errors should occur


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_client_get_session_id(mock_anthropic, tmp_path):
    """Test getting session ID from client."""
    mock_client = AsyncMock()
    mock_anthropic.return_value = mock_client

    options = ClaudeAgentOptions(
        session_store=FileSessionStore(base_path=tmp_path)
    )

    client = ClaudeSDKClient(options=options)
    session_id = await client.get_session_id()

    assert session_id is not None
    assert isinstance(session_id, str)
    assert len(session_id) > 0


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_query_without_options(mock_anthropic, tmp_path):
    """Test that query() works without explicit options."""
    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.create.return_value = mock_anthropic_stream("Response")
    mock_anthropic.return_value = mock_client

    # Patch the default FileSessionStore location
    with patch('claude_agent_sdk.api.FileSessionStore') as mock_file_store:
        mock_store_instance = AsyncMock()
        mock_file_store.return_value = mock_store_instance

        # Configure mock store
        mock_store_instance.load_session = AsyncMock(return_value=None)
        mock_store_instance.create_session = AsyncMock()
        mock_store_instance.append_message = AsyncMock()
        mock_store_instance.update_metadata = AsyncMock()

        # Should use default options
        async for _ in query("Test"):
            pass

        # Verify FileSessionStore was instantiated
        mock_file_store.assert_called_once()


@pytest.mark.asyncio
async def test_api_exports():
    """Test that public API exports are available."""
    from claude_agent_sdk import query, ClaudeSDKClient, ClaudeAgentOptions
    from claude_agent_sdk import FileSessionStore, RedisSessionStore
    from claude_agent_sdk import SessionStore, SessionState

    # All key exports should be available
    assert query is not None
    assert ClaudeSDKClient is not None
    assert ClaudeAgentOptions is not None
    assert FileSessionStore is not None
    assert RedisSessionStore is not None
```

4. **Create example usage documentation** in `tests/test_examples.py`:

```python
"""Example usage patterns for documentation."""

import pytest
from unittest.mock import patch, AsyncMock


def mock_stream(text: str):
    """Mock helper."""
    class MockBlock:
        type = "text"
    class MockDelta:
        type = "text_delta"
        def __init__(self, t):
            self.text = t
    class MockEvent:
        def __init__(self, event_type, **kwargs):
            self.type = event_type
            for k, v in kwargs.items():
                setattr(self, k, v)
    async def stream():
        yield MockEvent("content_block_start", content_block=MockBlock())
        yield MockEvent("content_block_delta", delta=MockDelta(text))
        yield MockEvent("message_stop")
    return stream()


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_example_simple_query(mock_anthropic):
    """Example: Simple one-off query."""
    from claude_agent_sdk import query

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.create.return_value = mock_stream("Hello!")
    mock_anthropic.return_value = mock_client

    # Simple query
    async for message in query("Hello, Claude!"):
        if message.role == "assistant":
            print(f"Claude says: {message.content}")


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_example_multi_turn_conversation(mock_anthropic, tmp_path):
    """Example: Multi-turn conversation."""
    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    from claude_agent_sdk.stores import FileSessionStore

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_anthropic.return_value = mock_client

    # Create client for conversation
    options = ClaudeAgentOptions(
        session_store=FileSessionStore(base_path=tmp_path)
    )

    async with ClaudeSDKClient(options=options) as client:
        # First question
        mock_client.messages.create.return_value = mock_stream("2+2 is 4")
        async for message in client.query("What's 2+2?"):
            pass

        # Follow-up question (uses same session)
        mock_client.messages.create.return_value = mock_stream("4*3 is 12")
        async for message in client.query("What's that times 3?"):
            pass


@pytest.mark.asyncio
@patch('claude_agent_sdk.api.AsyncAnthropic')
async def test_example_redis_backend(mock_anthropic):
    """Example: Using Redis storage backend."""
    from claude_agent_sdk import query, ClaudeAgentOptions
    from claude_agent_sdk.stores import RedisSessionStore

    mock_client = AsyncMock()
    mock_client.messages.create = AsyncMock()
    mock_client.messages.create.return_value = mock_stream("Response")
    mock_anthropic.return_value = mock_client

    # Use Redis for distributed deployment
    with patch('claude_agent_sdk.stores.redis_store.redis.from_url') as mock_redis:
        mock_redis.return_value = AsyncMock()

        options = ClaudeAgentOptions(
            session_store=RedisSessionStore("redis://localhost:6379")
        )

        async for message in query("Hello", options=options):
            pass
```

5. **Run all integration tests**:
   ```bash
   pytest tests/test_api.py tests/test_examples.py -v
   ```

6. **Update README.md** with usage examples:

```markdown
# Claude Agent SDK (Stateless)

Stateless Claude Agent SDK with pluggable storage backends.

## Installation

\`\`\`bash
pip install -e .

# Optional: Redis support
pip install -e ".[redis]"

# Optional: PostgreSQL support
pip install -e ".[postgres]"

# Development
pip install -e ".[dev]"
\`\`\`

## Quick Start

### Simple Query

\`\`\`python
from claude_agent_sdk import query

async for message in query("Hello, Claude!"):
    print(message.content)
\`\`\`

### Multi-Turn Conversation

\`\`\`python
from claude_agent_sdk import ClaudeSDKClient

async with ClaudeSDKClient() as client:
    await client.query("What's 2+2?")
    await client.query("What's that times 3?")
\`\`\`

### Redis Storage (Distributed)

\`\`\`python
from claude_agent_sdk import query, ClaudeAgentOptions
from claude_agent_sdk.stores import RedisSessionStore

options = ClaudeAgentOptions(
    session_store=RedisSessionStore("redis://localhost:6379")
)

async for message in query("Hello", options=options):
    print(message.content)
\`\`\`

## Architecture

See `ARCHITECTURE.md` for detailed design documentation.
\`\`\`

Ensure all tests pass and the public API is intuitive and well-documented.
</prompt>

---

