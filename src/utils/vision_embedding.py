"""Legacy model wrappers (Stripped during API transition).

This file previously contained heavy PyTorch/Transformers singletons.
It has been stripped to just provide stubs for backwards compatibility
in tests, as the application now uses vLLM and TEI via LangChain.
"""

import logging

logger = logging.getLogger(__name__)

# Stubs for backwards compatibility with tests


class Qwen3VisionEmbedder:
    def __init__(self, *args, **kwargs):
        pass

    def analyze_image(self, *args, **kwargs):
        return ""

    def cleanup(self):
        pass


class Qwen3TextLLM:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return ""


class Qwen3VLTextLLM:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return ""


class UnifiedQwen3VL:
    def __init__(self, *args, **kwargs):
        pass

    def analyze_image(self, *args, **kwargs):
        return ""

    def generate(self, *args, **kwargs):
        return ""

    def cleanup(self):
        pass


class UnifiedQwen35:
    def __init__(self, *args, **kwargs):
        pass

    def analyze_image(self, *args, **kwargs):
        return ""

    def generate(self, *args, **kwargs):
        return ""

    def cleanup(self):
        pass


class Qwen3VLEmbedding:
    def __init__(self, *args, **kwargs):
        pass

    def embed_text(self, *args, **kwargs):
        return []

    def embed_image(self, *args, **kwargs):
        return []

    def cleanup(self):
        pass


class Qwen3Embedding:
    def __init__(self, *args, **kwargs):
        pass

    def embed_text(self, *args, **kwargs):
        return []

    def embed_text_batch(self, *args, **kwargs):
        return []


class Qwen2TextLLM:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return ""


class Phi35TextLLM:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, *args, **kwargs):
        return ""


# Stubs for singletons
def reset_models():
    pass


def get_unified_model(*args, **kwargs):
    return UnifiedQwen35()


def get_vision_model(*args, **kwargs):
    return Qwen3VisionEmbedder()


def get_text_llm(*args, **kwargs):
    return Qwen3TextLLM()


def get_embedding_model(*args, **kwargs):
    return Qwen3Embedding()


def get_qwen2_llm(*args, **kwargs):
    return Qwen2TextLLM()


def get_phi35_llm(*args, **kwargs):
    return Phi35TextLLM()
