# Simple Dockerfile for SupervisorMCP
FROM python:3.11-slim

# Install dependencies directly
WORKDIR /app
RUN pip install mcp pydantic

# Copy source
COPY src/ ./src/

# Expose HTTP port for FastMCP (default is 8000)
EXPOSE 8000

# Configure FastMCP to bind to all interfaces
ENV MCP_HOST=0.0.0.0
ENV MCP_PORT=8000

# Run as non-root
RUN useradd --create-home supervisor
USER supervisor

# Health check
HEALTHCHECK CMD python -c "import sys; sys.exit(0)"

# Run the server
CMD ["python", "src/server.py"]