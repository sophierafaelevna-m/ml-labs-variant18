"""Lab 7 service extending the Lab 6 API."""

from src.lab6.service import app
from src.lab7.batch import router

app.title = "Lab 7 — Parallel Model Serving"
app.version = "2.0.0"

app.include_router(router)
