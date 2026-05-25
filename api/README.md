# FastAPI Local Endpoint

This folder contains a lightweight local API for the Bank Campaign Targeting System.

## Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/health` | Check whether the API is running |
| GET | `/policy` | Return the selected top-k targeting policy |
| POST | `/predict` | Score one customer and return a contact recommendation |

## Run locally

From the project root:

```bash
uvicorn api.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000/docs
```

## Test prediction

```bash
curl -X POST "http://127.0.0.1:8000/predict"   -H "Content-Type: application/json"   -d @api/sample_request.json
```

## Notes

This API is a local portfolio demonstration. It is not a production-ready banking API or real customer decisioning tool.
