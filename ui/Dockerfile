FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive
RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app
# Copy the entire UI structure
COPY streamlit_app.py ui_config.py ./
COPY utils/ ./utils/
COPY components/ ./components/
COPY static/ ./static/

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
