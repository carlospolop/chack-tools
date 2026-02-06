# chack-agent

A robust, configurable AI agent runtime with specialized toolsets for broad web research, scientific discovery, and social media intelligence. Built on top of `openai-agents` and designed for complex, multi-turn investigations.

## Installation

Install via pip:

```bash
pip install chack-agent
```

Or specific dependencies:

```bash
# Core only
pip install chack-agent

# With OpenAI Agents SDK support (required for sub-agents)
pip install "chack-agent[openai_agents]"
```

## Quick Start

```python
import os
from chack_agent import (
    Chack,
    ChackConfig,
    ModelConfig,
    AgentConfig,
    SessionConfig,
    ToolsConfig,
    CredentialsConfig,
    LoggingConfig,
)

# 1. Configure the agent
config = ChackConfig(
    model=ModelConfig(
        primary="gpt-4o",
        social_network="gpt-4o",  # Model for the social sub-agent
        scientific="gpt-4o",     # Model for the scientific sub-agent
        websearcher="gpt-4o",    # Model for the web search sub-agent
    ),
    agent=AgentConfig(
        self_critique_enabled=True,  # Agent critiques its own plan before acting
    ),
    session=SessionConfig(
        max_turns=30,
        memory_max_messages=20,          # Short-term context window
        long_term_memory_enabled=True,   # Enable file-based long-term memory
        long_term_memory_dir="./memory", # Where to store session summaries
    ),
    tools=ToolsConfig(
        exec_enabled=True,           # Allow running shell commands
        brave_enabled=True,          # Enable Brave Search
        websearcher_enabled=True,    # Enable the Web Researcher sub-agent
        scientific_enabled=True,     # Enable the Scientific Research sub-agent
        social_network_enabled=True, # Enable the Social Network sub-agent
    ),
    credentials=CredentialsConfig(
        openai_api_key=os.environ.get("OPENAI_API_KEY"),
    ),
    logging=LoggingConfig(level="INFO"),
    system_prompt="You are an advanced researcher agent.",
    env={},
)

# 2. Initialize and run
agent = Chack(config)
result = agent.run(
    session_id="investigation-001",
    text="Find recent research on plastic-eating bacteria and what people are saying about it on Reddit."
)

print(result.output)
```

## Key Features

### 1. Specialized Sub-Agents
The agent can delegate complex tasks to specialized sub-agents. These sub-agents run as independent loops with restricted toolsets to reduce noise and hallucination.

*   **Web Search Agent**: Performs deep research using Brave, Google, and Bing (including AI-mode summaries).
*   **Scientific Agent**: Searches academic sources including ArXiv, Europe PMC, Semantic Scholar, OpenAlex, PLOS, Google Scholar, and Google Patents. It can also transcribe YouTube videos and read PDFs.
*   **Social Network Agent**: Uses **ForumScout** tools to search LinkedIn, Instagram, Reddit, X (Twitter), and Google Forums/News.

### 2. Tool Ecosystem
The `ToolsConfig` object allows granular control over every tool:

*   **System Tools**:
    *   `exec`: Execute local shell commands (safe-guarded by timeout and output limits).
    *   `pdf_text`: Extract text from PDF files.
    *   `task_list`: Maintain a dynamic todo list for complex multi-step tasks.
*   **Web Tools**:
    *   `brave_search`: Privacy-focused web search.
    *   `serpapi`: Access to Google/Bing search results.

### 3. Memory Architecture
*   **Short-Term Memory**: Managed via `memory_max_messages` in `SessionConfig`. Keeps the immediate context window efficient.
*   **Long-Term Memory**: File-based persistence. The agent reads/writes summaries to a `long_term_memory_dir`. This allows it to recall key facts across different runs of the same `session_id`.

## Configuration & Environment Variables

Most tools require API keys. You can pass them into `CredentialsConfig` or `ToolsConfig`, or set them as environment variables.

| Environment Variable | Description | Required For |
|----------------------|-------------|--------------|
| `OPENAI_API_KEY` | OpenAI API Key | Core functionality |
| `BRAVE_API_KEY` | Brave Search API Key | `brave_search` tool |
| `SERPAPI_API_KEY` | SerpAPI Key | Google/Bing search, Google Scholar, Patents, YouTube |
| `FORUMSCOUT_API_KEY` | ForumScout API Key | Social Network sub-agent tools |
| `CHACK_EXEC_TIMEOUT` | Shell command timeout (sec) | `exec` tool (default: 120) |

### Detailed Config Structure

*   **`ModelConfig`**:
    *   `primary`: Main model for the coordinator agent.
    *   `*_max_turns`: Turn limits for specific sub-agents.
*   **`AgentConfig`**:
    *   `self_critique_enabled`: If true, the agent reflects on its plan before executing tools.
*   **`ToolsConfig`**:
    *   `exec_enabled`: **(Security Warning)** Allows the agent to run shell commands.
    *   `scientific_*_enabled`: Toggle specific scientific data sources.
    *   `social_network_*_enabled`: Toggle specific social platforms.

## Development

**Project Structure**:
*   `chack_agent/`: Core runtime, memory management, and agent logic.
*   `chack_tools/`: Tool implementations and sub-agent definitions.

**Running Tests**:
```bash
# Run verifying import of the toolset
python3 -c "from chack_tools.agents_toolset import AgentsToolset; print('Import OK')"
```
