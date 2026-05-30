# Backend deployment and secrets

Important notes to avoid leaking API keys and to deploy safely.

- **Never commit `.env` files**. The repository includes `backend/.env.example` as a template. Copy it to `backend/.env` locally and fill in your secrets.
- Add `backend/.env` to `.gitignore` (already added).

Quick local setup:

```bash
# copy example and edit
cp backend/.env.example backend/.env
# edit backend/.env and add your GROQ_API_KEY (do NOT commit)
```

Railway deployment (recommended):

- In the Railway project, set the environment variable `GROQ_API_KEY` using Railway Secrets/Environment UI.
- Set the service root to the `backend/` folder and the start command to:

```bash
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

Prevent accidental commits of secrets:

1. Enable the provided git hook to block obvious secrets in staged files:

```bash
git config core.hooksPath .githooks
chmod +x .githooks/pre-commit
```

2. If you accidentally committed a secret, rotate the key immediately and remove it from git history. At minimum:

```bash
git rm --cached backend/.env
git commit -m "Remove backend .env from repo"
git push
```

For full removal from history use `git filter-repo` or the BFG Repo-Cleaner; follow their docs.

If you want, I can also:

- Add a GitHub Actions secret scan workflow.
- Run commands to remove an already-committed secret (you must confirm and provide no secret values here).
