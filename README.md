# SupervisorMCP - Simple AI Agent Supervisor

A minimal Model Context Protocol (MCP) server that provides basic supervision for AI agents.

## What it does

- **start_job**: Break down work into simple tasks
- **task_update**: Track progress and give basic feedback  
- **report_problem**: Get simple solutions for common problems

## Quick Start

```bash
# Build and run
docker build -t supervisor-mcp .
docker run -it supervisor-mcp

# Or locally
pip install -e .
python src/server.py
```

## Usage

Use with VS Code Chat or any MCP client:

```
@supervisor I need to build a web API with authentication
```

That's it. Simple, focused, works.
