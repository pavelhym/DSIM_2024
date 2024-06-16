FROM python:3.10

EXPOSE 8506

WORKDIR /app

COPY requirements.txt ./requirements.txt

# Update package lists and install dependencies
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt



COPY . .

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8506"]