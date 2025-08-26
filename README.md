# SupervisorMCP

An MCP server that provides intelligent task management for AI agents using OpenAI for job breakdown, progress tracking, and problem analysis.
Why is this useful? If you have an agent - coding agent, browser use agent etc - doing a long running or complex task, you can get it to start the job via the MCP supervisor, and update the tasks as it works.
If the agent runs into any errors, or needs to backtrack, it has a consistent external list of whats been done or not.
Try messages like "check job status in mcp supervisor, confirm in codebase if changes already made, if so, mark tasks complete, if not, keep working on it"
This works especially well if you empower the agent to constructively disagree with the supervisor, or close irrelevant tasks as it works.
You can also use this across sessions, breaking changes up into multiple tasks, and creating new chats between them, to keep context smaller.

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
# Docker (recommended)
docker build -t supervisor-mcp .
docker run -d -e OPENAI_API_KEY="your-api-key" supervisor-mcp
```

Adding the MCP server:
```json
{
    "servers": {
        "MCP_SUPERVISOR": {
            "url": "http://localhost:8000/mcp",
            "headers": {"Content-Type": "application/json"},
            "type": "http"
        }
    }
}
```

## Requirements

- Python 3.11+
- OpenAI API key
- Dependencies: `mcp`, `pydantic`, `openai`

## Testing

See [test-script.md](./test-script.md) for comprehensive test procedures covering all functionality - can be run by your coding agent as a "test-suite-lite" for the MCP server during development, or if you want a demonstration of it's capabilities.
