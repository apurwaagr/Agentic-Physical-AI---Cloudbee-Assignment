# Dockerfile  —  Task 3B
# ─────────────────────────────────────────────────────────────────────────────
# Must work with:
#   docker build -t scene-gen .
#   docker run -p 8000:8000 scene-gen
#
# The app must be reachable at http://localhost:8000/generate after docker run.
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install only the packages needed at runtime (no torch/lerobot needed here)
COPY part3/requirements_api.txt .
RUN pip install --no-cache-dir -r requirements_api.txt

# Copy application code
# part2/ provides scene_generator.py; part3/ provides app.py
COPY part2/ ./part2/
COPY part3/ ./part3/

# Expose the API port
EXPOSE 8000

# Start the FastAPI server
CMD ["uvicorn", "part3.app:app", "--host", "0.0.0.0", "--port", "8000"]

