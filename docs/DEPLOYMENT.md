# Production deployment

This repository deploys as two services:

* **Vercel** hosts the Vite/React frontend.
* **Render** hosts the FastAPI API from `backend/`, using the included `render.yaml` Blueprint.
* **MongoDB Atlas** persists users, uploaded data, documents and investigations. The default JSON store is only for local development.

## Required configuration

Create a MongoDB Atlas database user and obtain its SRV connection string. In the Render service, set the following environment variables:

```text
MONGODB_URI=mongodb+srv://...
MONGODB_DB=businessintelligence
FIREBASE_PROJECT_ID=your-firebase-project-id
FIREBASE_SERVICE_ACCOUNT_JSON={single-line Firebase service account JSON}
CORS_ORIGINS=https://your-vercel-project.vercel.app
ANTHROPIC_API_KEY=...                       # optional; deterministic fallback works without it
```

The Blueprint supplies `APP_ENV=production`, `DB_BACKEND=mongo`, and `AUTH_MODE=firebase`.

In Vercel, set:

```text
VITE_API_BASE_URL=https://your-render-service.onrender.com
VITE_FIREBASE_API_KEY=...
VITE_FIREBASE_AUTH_DOMAIN=...
VITE_FIREBASE_PROJECT_ID=...
VITE_FIREBASE_STORAGE_BUCKET=...
VITE_FIREBASE_MESSAGING_SENDER_ID=...
VITE_FIREBASE_APP_ID=...
```

After the frontend URL is available, add that hostname in Firebase Authentication's **Authorised domains** and set the exact same URL in Render's `CORS_ORIGINS`. Redeploy Vercel after changing any `VITE_*` setting because Vite embeds them at build time.

## Verification

1. Visit `https://your-render-service.onrender.com/api/health`; it should return `status: ok`.
2. Visit `/api/system/status`; verify `database.backend` is `mongo` and `auth.mode` is `firebase`.
3. Sign up with Firebase on the Vercel URL and run the sample-data investigation.
