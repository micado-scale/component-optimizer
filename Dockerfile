FROM python:3.7.4

WORKDIR /optimizer

COPY . .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

ENTRYPOINT ["./docker-entrypoint.sh"]

CMD ["--cfg", "config/config.yaml", "--host", "0.0.0.0", "--port", "5000"]
