FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
# pyarrow is an OPTIONAL accelerator: the tool auto-detects it for a much faster
# Arrow-based loader, and falls back to the pandas loader when it is absent.
# Installed here so the official image is fast; not added to requirements.txt so
# pip/conda installs stay lightweight unless the user opts in.
RUN pip install --no-cache-dir pyarrow
COPY cgmlst-dists.py /app/
COPY entrypoint.sh /app/
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
