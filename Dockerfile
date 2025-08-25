# Simple Dockerfile for SupervisorMCP
FROM python:3.11-slim

# Install dependencies directly
WORKDIR /app
RUN python -m pip install mcp pydantic openai

# Copy source code
COPY src/ ./src/

# Expose HTTP port for FastMCP (default is 8000)
EXPOSE 8000

# Configure FastMCP to bind to all interfaces
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000
a
# Run the server
CMD ["python", "src/server.py"]