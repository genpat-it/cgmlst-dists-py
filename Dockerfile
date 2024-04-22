FROM python:3.9
WORKDIR /app
COPY cgmlst-dists.py /app/
COPY entrypoint.sh /app/
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Set the entrypoint script to be executed
ENTRYPOINT ["/app/entrypoint.sh"]
