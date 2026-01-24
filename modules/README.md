# Docker Compose Stack Overview

This repository includes a `docker-compose.yml` that spins up a small local stack for an object-detection / LLM-assisted app:

- **PostgreSQL** for structured data
- **Prisma migrations** runner for the data layer
- **LocalStack (S3)** for local object storage
- **Chainlit-based app** (served on port 8080)
- **Ollama adapter** (HTTP wrapper/proxy to an Ollama instance running on your host)

This setup is intended for **local development**. The credentials in the compose file are defaults for convenience—change them for anything beyond local use.

---



## Volumes (Persistence)

Used volumes:

- `obj-det-pg` — PostgreSQL data directory
- `obj-det-s3-vol` — LocalStack persistent state

Declared (but not currently used by ollama in this compose file):

- `ollama-data`

Need for ollama service that currently commented in compose file.

---

