# SupervisorMCP

An MCP server that provides intelligent task management for AI agents using OpenAI for job breakdown, progress tracking, and problem analysis.

## Features

- **start_job** - Breaks complex jobs into actionable tasks using LLM analysis
- **update_task** - Tracks progress with intelligent feedback generation  
- **complete_task** - Marks tasks complete with next-step recommendations
- **report_problem** - Analyzes problems using multi-perspective LLM approach
- **get_all_jobs** / **get_job_tasks** - Retrieves job and task data
- **prune_job** - Deletes jobs and associated tasks

## Architecture

- **Clean separation**: Server, service, storage, and LLM client layers
- **Data persistence**: JSON Lines storage for jobs and tasks
- **Structured outputs**: Pydantic models with OpenAI structured completion
- **Error handling**: Graceful failure modes with meaningful messages

## Setup

```bash
# Set OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Docker (recommended)
docker build -t supervisor-mcp .
docker run -it supervisor-mcp

# Local development
python src/server.py
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies: `mcp`, `pydantic`, `openai`

## Testing

See [test-script.md](./test-script.md) for comprehensive test procedures covering all functionality.
