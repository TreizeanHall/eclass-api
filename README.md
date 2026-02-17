# EClass API (Email Classifier)

Dockerized FastAPI service for email classification.

---

## Quickstart (Docker)

### Build the image

```bash
docker build -t eclass-api:local .
```

---

### Run the container

```bash
docker run --rm -p 8000:8000 -e API_KEY="devkey123" eclass-api:local
```

The API will start at:

```
http://localhost:8000
```

---

## Health Check

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{"status":"ok"}
```

---

## Predict an Email Label

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"subject":"Need COI","description":"Please send certificate for landlord"}'
```

---

## Train the Model (Protected Endpoint)

Requires API key header.

```bash
curl -X POST "http://localhost:8000/train" \
  -H "Authorization: Bearer devkey123" \
  -H "Content-Type: application/json" \
  -d "{}"
```

---

##  Environment Variables

| Variable | Required | Description                            |
| -------- | -------- | -------------------------------------- |
| API_KEY  | Yes      | Protects admin endpoints like `/train` |

---

##  Endpoints Summary

| Method | Endpoint   | Description                    |
| ------ | ---------- | ------------------------------ |
| GET    | `/health`  | Service health check           |
| POST   | `/predict` | Predict classification label   |
| POST   | `/train`   | Train model (requires API key) |

---

## Notes for Cloud Deployment

* The service is fully containerized and ready for Azure deployment.
* Model artifacts are stored locally in `/artifacts/latest` during training.
* In production, artifacts should be persisted using Azure Blob Storage or Azure Files.

---

## Maintainer
Treizean Hall

