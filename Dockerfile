# Step 1: Python ka base image istemal karein
FROM python:3.12-slim

# Step 2: Container ke andar kaam karne ke liye ek directory banayein
WORKDIR /app

# Step 3: Sirf requirements file ko pehle copy karein
COPY requirements.txt .

# Step 4: Dependencies install karein aur Spacy model download karein
RUN pip install --no-cache-dir -r requirements.txt && \
    python -m spacy download en_core_web_md

# Step 5: Baaqi tamam application code ko copy karein
COPY . .

# Step 6: Fly.io ko batayein ke app port 8080 par chalegi
EXPOSE 8080

# Step 7: Application ko Gunicorn ke zariye start karein
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]
