# Backend Deployment

This folder contains the ATVDCS FastAPI backend.

## Build locally with Docker

```bash
cd atvdcs_v2
docker build -t atvdcs-backend .
```

## Run locally

```bash
docker run --rm -p 8000:8000 atvdcs-backend
```

Then open http://localhost:8000/health to verify the backend is running.

## Recommended production deployment

The backend is container-ready and can be deployed to any container host such as:
- Railway
- Render
- Fly.io
- Google Cloud Run
- AWS App Runner
- Azure App Service for Containers

## Deploying to Render

1. Push this repository to GitHub.
2. In Render, create a new Web Service and connect your GitHub repo.
3. Use the Docker deployment option and set the Dockerfile path to `atvdcs_v2/Dockerfile`.
4. Set the root directory to `/` and deploy from branch `main`.
5. The service should start with `uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}`.
6. Once deployed, your backend URL will typically be `https://ai-traffic-backend.onrender.com` if that name is available.

## Notes

- The service exposes `uvicorn api:app --host 0.0.0.0 --port 8000`.
- CORS is already configured to allow requests from your frontend.
- The app expects `config/config.yaml`, `yolov8n.pt`, and the `modules/` package to be present.
- If you deploy to a managed container host, set the port to `8000` or map the host's port to `8000`.

## If you want a GitHub Actions deployment

I can also add a workflow that builds the container and pushes it to a registry once you share the target provider and credentials.
