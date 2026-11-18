# Genesis Workbench — React + FastAPI app

The Databricks App that the bundle deploys as `genesis-workbench`. FastAPI
serves the JSON API at `/api/*` and the built React SPA at `/`.

## Layout

```
app/
├── app.yml                  # Databricks Apps launch command + env (uvicorn)
├── requirements.txt         # Python deps installed by Databricks Apps
├── backend/
│   ├── lib/                 # genesis_workbench wheel (rebuilt + staged by core/deploy.sh)
│   ├── requirements.txt     # dev-only deps (used for the local venv)
│   ├── documentation/       # per-workflow reference docs the assistant pulls into prompts
│   └── app/
│       ├── main.py          # FastAPI app, lifespan hook, static-SPA fallback
│       ├── auth.py          # Depends() that builds WorkspaceClient from X-Forwarded-Access-Token
│       ├── config.py        # env-var-backed Settings dataclass
│       ├── routers/         # API surface — one router per page (large_molecule, small_molecule,
│       │                    #   genomics, single_cell, models, settings, monitoring, …)
│       └── services/        # business logic — Databricks SDK calls, MLflow searches, Mol* HTML
└── frontend/
    ├── .npmrc               # pinned to https://npm-proxy.cloud.databricks.com/
    ├── package.json
    ├── vite.config.ts       # dev proxy /api → :8000; build outputs to ./dist
    ├── tailwind.config.ts
    └── src/
        ├── main.tsx
        ├── App.tsx          # Router + QueryClient
        ├── components/      # shared UI primitives (Layout, Tabs, DataTable, dialogs)
        ├── pages/           # one page per workflow module (LargeMolecule, Genomics, …)
        ├── stores/          # Zustand stores (user, theme)
        ├── api/             # typed REST client
        ├── types/           # API response types
        └── lib/utils.ts
```

## Local dev loop

Two terminals:

```bash
# Terminal 1 — backend
cd modules/core/app/backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (Vite HMR, proxies /api/* to :8000)
cd modules/core/app/frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Anything under `/api/me`, `/api/bootstrap`,
or workflow streaming routes will return 401/empty locally — they expect the
`X-Forwarded-Access-Token` header that only Databricks Apps SSO injects.
`/api/health` works without auth and is the fastest smoke test.

## Deploy

Use `./deploy.sh core <cloud>` from the repo root — it runs `npm install` /
`npm run build` and then `databricks bundle deploy`, which uploads the source
code (including the built `frontend/dist/`) and registers the app under the
name `genesis-workbench`. The bundle's
[`sync.include`](https://github.com/databricks-industry-solutions/genesis-workbench/blob/HEAD/modules/core/databricks.yml) directive force-uploads `frontend/dist/**`
even though Vite's gitignore excludes it.

## Codegen (typed API client)

`src/types/api.ts` is hand-maintained today. To regenerate from the live
OpenAPI schema once the backend is running on port 8000:

```bash
cd modules/core/app/frontend
npm run gen:api
```
