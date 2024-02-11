FROM python:3.10

EXPOSE 8506

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install -r requirements.txt

COPY . .

ENTRYPOINT ["streamlit", "run", "dashboard.py", "--server.port=8506"]