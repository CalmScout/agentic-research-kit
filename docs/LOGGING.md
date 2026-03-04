# Logging Infrastructure

## Overview

The Agentic Research Kit (ARK) uses a unified, structured logging system powered by [Loguru](https://github.com/Delgan/loguru). It is designed to provide high visibility during development while being robust enough for production monitoring.

### Key Features
- 🧩 **Unified Interface**: Single logger instance used across all modules (`from src.utils.logger import logger`).
- 📁 **Dual Sinks**: Simultaneous logging to console and persistent files.
- 🔄 **Automatic Management**: Log rotation (10 MB) and retention (30 days) with ZIP compression.
- 🏗️ **Structured Formats**: Support for human-readable text (development) and machine-parseable JSON (production).
- ⚠️ **Error Isolation**: Dedicated `ark_errors.log` for high-priority troubleshooting.

---

## Configuration

Logging behavior is controlled via environment variables through the `Settings` class:

| Variable | Description | Options | Default |
|----------|-------------|---------|---------|
| `LOG_LEVEL` | Minimum severity level | `DEBUG`, `INFO`, `WARNING`, `ERROR` | `INFO` |
| `LOG_FORMAT`| Console output format | `text`, `json` | `json` |
| `LOG_DIR`   | Directory for log files| Path string | `./logs` |

### Example: Development Mode
```bash
LOG_LEVEL=DEBUG LOG_FORMAT=text uv run ark query "..."
```

### Example: Production Mode
```bash
LOG_LEVEL=INFO LOG_FORMAT=json uv run ark serve
```

---

## Log Output

### 1. Console (Text Mode)
The text format is color-coded and includes the timestamp, level, and location:
`2026-03-04 15:35:11 | INFO | src.agents.workflow:query_with_agents:157 - ✓ Query complete`

### 2. Files
Logs are stored in the directory specified by `LOG_DIR`:
- `ark_YYYY-MM-DD.log`: Full logs for the day (respects `LOG_LEVEL`).
- `ark_errors.log`: Filtered log containing only `ERROR` level events and above.

---

## Usage in Code

To ensure consistent formatting and file persistence, always import the logger from the project utility:

```python
from src.utils.logger import logger

def my_function():
    logger.info("Function started", detail="some metadata")
    try:
        # logic
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
```

---

**Last Updated**: 2026-03-04
