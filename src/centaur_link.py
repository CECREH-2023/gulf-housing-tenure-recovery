"""CENTAUR infrastructure linkage module.

This module handles the connection between Gulf_Hazard_Interview
and the CENTAUR infrastructure.

It supports a hybrid setup:
- centaur-platform: For multilanguage/R support
- ai-disaster-agent: For RAG and embedding components

Usage:
    from centaur_link import get_llm_provider, get_embedding_service, create_vector_store

    # Get LLM provider
    provider = get_llm_provider()

    # Get embedding service
    embedder = get_embedding_service()

    # Create vector store
    store = create_vector_store(dimensions=384)

Environment Variables:
    CENTAUR_PATH: Primary CENTAUR installation path
    CENTAUR_RAG_PATH: Path to CENTAUR with RAG components (ai-disaster-agent)
"""

import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

# Default CENTAUR locations (configurable via environment)
# centaur-platform: multilanguage/R support
# ai-disaster-agent: RAG and embedding components
DEFAULT_CENTAUR_PATH = Path(__file__).parent.parent / "centaur-platform"
DEFAULT_RAG_PATH = Path(__file__).parent.parent / "ai-disaster-agent"

CENTAUR_PATH = Path(os.environ.get("CENTAUR_PATH", str(DEFAULT_CENTAUR_PATH)))
CENTAUR_RAG_PATH = Path(os.environ.get("CENTAUR_RAG_PATH", str(DEFAULT_RAG_PATH)))

# Track if CENTAUR has been initialized
_centaur_initialized = False
_rag_initialized = False


class CENTAURNotFoundError(RuntimeError):
    """Raised when CENTAUR installation is not accessible."""
    pass


class CENTAURImportError(ImportError):
    """Raised when a CENTAUR component cannot be imported."""
    pass


def ensure_centaur_available() -> bool:
    """Ensure primary CENTAUR (centaur-platform) is accessible.

    Returns:
        True if CENTAUR is available and initialized.

    Raises:
        CENTAURNotFoundError: If CENTAUR installation not found.
    """
    global _centaur_initialized

    if _centaur_initialized:
        return True

    if not CENTAUR_PATH.exists():
        raise CENTAURNotFoundError(
            f"CENTAUR not found at {CENTAUR_PATH}. "
            f"Set CENTAUR_PATH environment variable to the correct location."
        )

    # Verify key CENTAUR directories exist (relaxed - centaur-platform may not have all)
    required_dirs = ["src/llm"]
    for dir_name in required_dirs:
        if not (CENTAUR_PATH / dir_name).exists():
            raise CENTAURNotFoundError(
                f"CENTAUR installation appears incomplete: {dir_name} not found"
            )

    # Add CENTAUR to Python path
    centaur_str = str(CENTAUR_PATH)
    if centaur_str not in sys.path:
        sys.path.insert(0, centaur_str)

    _centaur_initialized = True
    return True


def ensure_rag_available() -> bool:
    """Ensure RAG-enabled CENTAUR (ai-disaster-agent) is accessible.

    Returns:
        True if RAG components are available and initialized.

    Raises:
        CENTAURNotFoundError: If RAG installation not found.
    """
    global _rag_initialized

    if _rag_initialized:
        return True

    if not CENTAUR_RAG_PATH.exists():
        raise CENTAURNotFoundError(
            f"CENTAUR RAG components not found at {CENTAUR_RAG_PATH}. "
            f"Set CENTAUR_RAG_PATH environment variable to the correct location."
        )

    # Verify RAG directory exists
    if not (CENTAUR_RAG_PATH / "src" / "rag").exists():
        raise CENTAURNotFoundError(
            f"CENTAUR RAG module not found at {CENTAUR_RAG_PATH}/src/rag"
        )

    # Add to Python path
    rag_str = str(CENTAUR_RAG_PATH)
    if rag_str not in sys.path:
        sys.path.insert(0, rag_str)

    _rag_initialized = True
    return True


def get_centaur_component(component: str) -> Any:
    """Import a CENTAUR component dynamically.

    Args:
        component: Dot-separated path (e.g., "src.llm.get_llm_provider")

    Returns:
        The imported component.

    Raises:
        CENTAURImportError: If the component cannot be imported.
    """
    ensure_centaur_available()

    try:
        parts = component.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attr_name = parts
            module = __import__(module_path, fromlist=[attr_name])
            return getattr(module, attr_name)
        else:
            return __import__(component)
    except (ImportError, AttributeError) as e:
        raise CENTAURImportError(f"Failed to import CENTAUR component '{component}': {e}")


def _load_module_from_file(module_path: str, base_path: Path) -> Any:
    """Load a module directly from file, caching in sys.modules.

    Args:
        module_path: Dot-separated module path (e.g., "src.llm.provider")
        base_path: Base path for module resolution

    Returns:
        The loaded module.
    """
    if module_path in sys.modules:
        return sys.modules[module_path]

    file_path = base_path / (module_path.replace(".", "/") + ".py")

    if not file_path.exists():
        raise ImportError(f"Module file not found: {file_path}")

    spec = importlib.util.spec_from_file_location(module_path, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_path] = module
    spec.loader.exec_module(module)
    return module


def get_rag_component(component: str) -> Any:
    """Import a RAG component from ai-disaster-agent.

    Uses importlib to directly load modules, bypassing potentially broken
    package __init__.py files. Handles module dependencies automatically.

    Args:
        component: Dot-separated path (e.g., "src.rag.vector_store.FAISSVectorStore")

    Returns:
        The imported component.

    Raises:
        CENTAURImportError: If the component cannot be imported.
    """
    ensure_rag_available()

    # Module dependencies that need to be pre-loaded
    dependencies = {
        "src.llm.sentence_transformers_provider": ["src.llm.provider"],
        "src.rag.embeddings": ["src.llm.provider"],
    }

    try:
        parts = component.rsplit(".", 1)
        if len(parts) == 2:
            module_path, attr_name = parts

            # Pre-load dependencies if needed
            if module_path in dependencies:
                for dep in dependencies[module_path]:
                    _load_module_from_file(dep, CENTAUR_RAG_PATH)

            # Convert module path to file path
            file_path = CENTAUR_RAG_PATH / (module_path.replace(".", "/") + ".py")

            if file_path.exists():
                module = _load_module_from_file(module_path, CENTAUR_RAG_PATH)
                return getattr(module, attr_name)
            else:
                # Fallback to standard import
                module = __import__(module_path, fromlist=[attr_name])
                return getattr(module, attr_name)
        else:
            return __import__(component)
    except (ImportError, AttributeError) as e:
        raise CENTAURImportError(f"Failed to import RAG component '{component}': {e}")


# =============================================================================
# LLM Provider Interface
# =============================================================================

@lru_cache(maxsize=1)
def get_llm_provider(provider_type: Optional[str] = None):
    """Get CENTAUR LLM provider.

    Args:
        provider_type: Optional provider type ("anthropic", "openai", "ollama").
                      If None, uses default from CENTAUR settings.

    Returns:
        An LLM provider instance.
    """
    get_provider = get_centaur_component("src.llm.get_llm_provider")
    if provider_type:
        return get_provider(provider_type)
    return get_provider()


def get_embedding_provider(model_name: Optional[str] = None):
    """Get CENTAUR embedding provider.

    Args:
        model_name: Optional model name (e.g., "all-MiniLM-L6-v2").
                   If None, uses default from CENTAUR settings.

    Returns:
        An embedding provider instance.
    """
    SentenceTransformersProvider = get_rag_component(
        "src.llm.sentence_transformers_provider.SentenceTransformersProvider"
    )
    model = model_name or "all-MiniLM-L6-v2"
    return SentenceTransformersProvider(model_name=model)


# =============================================================================
# RAG Components Interface
# =============================================================================

def create_vector_store(
    dimensions: int = 384,
    metric: str = "cosine",
    index_path: Optional[str] = None,
):
    """Create a CENTAUR FAISS vector store.

    Args:
        dimensions: Embedding dimensions (default 384 for MiniLM).
        metric: Distance metric ("cosine", "l2", "ip").
        index_path: Optional path to load existing index (base path without extension).

    Returns:
        A FAISSVectorStore instance.
    """
    FAISSVectorStore = get_rag_component("src.rag.vector_store.FAISSVectorStore")

    if index_path:
        # Check for the .faiss file (load expects base path, files are base.faiss and base.json)
        faiss_file = Path(str(index_path) + ".faiss")
        if faiss_file.exists():
            return FAISSVectorStore.load(index_path)

    return FAISSVectorStore(dimensions=dimensions, metric=metric)


def get_document_class():
    """Get the CENTAUR Document dataclass."""
    return get_rag_component("src.rag.vector_store.Document")


def get_search_result_class():
    """Get the CENTAUR SearchResult dataclass."""
    return get_rag_component("src.rag.vector_store.SearchResult")


# =============================================================================
# Message Classes Interface
# =============================================================================

def get_llm_message_class():
    """Get the CENTAUR LLMMessage dataclass."""
    return get_centaur_component("src.llm.provider.LLMMessage")


def get_llm_response_class():
    """Get the CENTAUR LLMResponse dataclass."""
    return get_centaur_component("src.llm.provider.LLMResponse")


def get_message_role_enum():
    """Get the CENTAUR MessageRole enum."""
    return get_centaur_component("src.llm.provider.MessageRole")


# =============================================================================
# Convenience Functions
# =============================================================================

def create_system_message(content: str):
    """Create a system message using CENTAUR's LLMMessage."""
    LLMMessage = get_llm_message_class()
    return LLMMessage.system(content)


def create_user_message(content: str):
    """Create a user message using CENTAUR's LLMMessage."""
    LLMMessage = get_llm_message_class()
    return LLMMessage.user(content)


def create_assistant_message(content: str):
    """Create an assistant message using CENTAUR's LLMMessage."""
    LLMMessage = get_llm_message_class()
    return LLMMessage.assistant(content)


# =============================================================================
# Health Check
# =============================================================================

def check_centaur_health() -> dict[str, Any]:
    """Check CENTAUR installation health.

    Returns:
        Dictionary with health check results for both centaur-platform and ai-disaster-agent.
    """
    results = {
        "centaur_path": str(CENTAUR_PATH),
        "rag_path": str(CENTAUR_RAG_PATH),
        "centaur_available": False,
        "rag_available": False,
        "components": {},
    }

    # Check centaur-platform (LLM/R support)
    try:
        ensure_centaur_available()
        results["centaur_available"] = True

        # Check LLM components from centaur-platform
        try:
            get_centaur_component("src.llm.provider.LLMProvider")
            results["components"]["llm_provider"] = "ok"
        except CENTAURImportError as e:
            results["components"]["llm_provider"] = f"error: {e}"

    except CENTAURNotFoundError as e:
        results["centaur_error"] = str(e)

    # Check ai-disaster-agent (RAG/embeddings)
    try:
        ensure_rag_available()
        results["rag_available"] = True

        # Check RAG components from ai-disaster-agent
        rag_components = [
            ("vector_store", "src.rag.vector_store.FAISSVectorStore"),
            ("document", "src.rag.vector_store.Document"),
            ("embeddings", "src.llm.sentence_transformers_provider.SentenceTransformersProvider"),
        ]

        for name, path in rag_components:
            try:
                get_rag_component(path)
                results["components"][name] = "ok"
            except CENTAURImportError as e:
                results["components"][name] = f"error: {e}"

    except CENTAURNotFoundError as e:
        results["rag_error"] = str(e)

    return results


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Initialization
    "ensure_centaur_available",
    "ensure_rag_available",
    "get_centaur_component",
    "get_rag_component",
    "CENTAUR_PATH",
    "CENTAUR_RAG_PATH",

    # LLM
    "get_llm_provider",
    "get_embedding_provider",

    # RAG
    "create_vector_store",
    "get_document_class",
    "get_search_result_class",

    # Messages
    "get_llm_message_class",
    "get_llm_response_class",
    "get_message_role_enum",
    "create_system_message",
    "create_user_message",
    "create_assistant_message",

    # Health
    "check_centaur_health",

    # Exceptions
    "CENTAURNotFoundError",
    "CENTAURImportError",
]
