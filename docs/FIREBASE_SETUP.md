# Firebase setup

**You do not need this to run the project.** With no Firebase configuration the backend runs
in demo mode: it enables a local login endpoint, states clearly in the UI that ID-token
signatures are not being verified, and everything else works exactly the same.

Follow this when you want real authentication.

---

## 1. Create the project

1. Go to <https://console.firebase.google.com> → **Add project**.
2. Name it (for example `businessintelligence-ai`). Google Analytics is not needed.

## 2. Enable sign-in methods

**Build → Authentication → Get started → Sign-in method**

* Enable **Email/Password**.
* Enable **Google** (optional; the app shows a "Continue with Google" button when Firebase is
  configured). Set a support email.

## 3. Register the web app → frontend keys

**Project settings (gear icon) → General → Your apps → Web (`</>`)**

Register the app (nickname `bi-ai-web`; Firebase Hosting not required). Copy the config into
`frontend/.env`:

```bash
# frontend/.env
VITE_API_BASE_URL=http://localhost:8000

VITE_FIREBASE_API_KEY=AIzaSy…
VITE_FIREBASE_AUTH_DOMAIN=businessintelligence-ai.firebaseapp.com
VITE_FIREBASE_PROJECT_ID=businessintelligence-ai
VITE_FIREBASE_STORAGE_BUCKET=businessintelligence-ai.appspot.com
VITE_FIREBASE_MESSAGING_SENDER_ID=123456789012
VITE_FIREBASE_APP_ID=1:123456789012:web:abc123def456
```

Restart `npm run dev` — Vite only reads `.env` at start-up.

> These values are **not secrets**. They identify your project in the browser; access is
> controlled by Firebase security rules and by our backend's token verification.

## 4. Service account → backend verification

**Project settings → Service accounts → Generate new private key** — this downloads a JSON
file. **This one is a secret. Never commit it.**

Either point at the file:

```bash
# backend/.env
FIREBASE_PROJECT_ID=businessintelligence-ai
GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/serviceAccountKey.json
AUTH_MODE=firebase
```

or paste the JSON as a single line (better for hosted deployments):

```bash
FIREBASE_PROJECT_ID=businessintelligence-ai
FIREBASE_SERVICE_ACCOUNT_JSON={"type":"service_account","project_id":"…","private_key":"-----BEGIN PRIVATE KEY-----\n…"}
AUTH_MODE=firebase
```

Restart the backend. On start-up you should see:

```
INFO app.auth.firebase_auth: Firebase Admin initialised — ID tokens will be verified.
```

## 5. Verify

```bash
curl -s localhost:8000/api/system/status | python3 -m json.tool
```

`auth.mode` should read `firebase` and the demo-mode notice should disappear from the sign-in
page. Once verification is active, demo tokens are rejected outright — the mode never silently
degrades.

## 6. Authorised domains (for deployment)

**Authentication → Settings → Authorised domains** — add the domain you deploy the frontend
to. `localhost` is authorised by default.

---

## How authentication and authorisation fit together

```
Browser                         Backend
───────                         ───────
Firebase JS SDK signs in
  → ID token (JWT)
  → Authorization: Bearer …  ─→  firebase-admin verifies the signature
                                 ↓
                                 uid, email
                                 ↓
                                 our user store → ROLE
                                 ↓
                                 role decides which fields the response contains
```

**Firebase answers "who are you?".** The role in our own database answers "what may you see?",
and the backend removes analyst-only fields from responses for other roles — see
`backend/app/api/redact.py`. Hiding fields in the interface alone would not be authorisation.

---

## `AUTH_MODE`

| Value | Behaviour |
|---|---|
| `auto` *(default)* | Verify when service-account credentials are present; demo mode otherwise |
| `firebase` | Always verify. Demo login is disabled and demo tokens are rejected |
| `demo` | Never verify. Local login enabled. Real Firebase tokens are still accepted but marked `verified: false` |

Use `firebase` for anything deployed.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| "This sign-in method is not enabled…" | Provider not enabled in the console | Authentication → Sign-in method |
| `401 invalid Firebase ID token` | Backend service account belongs to a different project | Match `FIREBASE_PROJECT_ID` to the web app's project |
| Demo notice still shows after configuring | Backend not restarted, or the credentials path is wrong | Restart; check `/api/system/status` |
| Google popup closes immediately | Domain not authorised | Authentication → Settings → Authorised domains |
| Frontend still in demo mode | Vite reads `.env` only at start-up | Restart `npm run dev` |
| `401` on every request after a long session | ID token expired | The SDK refreshes automatically; reload the page if it persists |

## Security notes

* `.env` files and service-account JSON are in `.gitignore`. Keep them there.
* The frontend config values are public by design; the service-account key is not.
* Rotate the service-account key if it is ever committed or shared.
* For production, restrict CORS: set `CORS_ORIGINS` in `backend/.env` to your deployed origin.
