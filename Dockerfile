FROM python:3.10.10-slim

WORKDIR C:\Users\elsem\Python\Bird_classification

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

# EXPOSE 8501

CMD streamlit run app.py 
