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

