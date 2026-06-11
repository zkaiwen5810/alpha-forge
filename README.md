# alpha-forge

## Local secrets

Store local API keys and other secrets in `.env.local`.

1. Copy `.env.example` to `.env.local`
2. Replace placeholder values with your real local secrets
3. Export `UV_ENV_FILE=.env.local`
4. Start a script with `uv run python example.py`

Example:

```env
OPENAI_API_KEY=your-real-key
GITHUB_TOKEN=your-real-token
```

Read values in Python with `os.getenv("OPENAI_API_KEY")`.

One-off run:

```sh
UV_ENV_FILE=.env.local uv run python example.py
```

Current shell session:

```sh
export UV_ENV_FILE=.env.local
uv run python example.py
```

If you want that to be automatic every time you enter the repo, use a shell tool like `direnv` to export `UV_ENV_FILE=.env.local` for this directory.

`.env.local` is git-ignored. Keep real secrets out of committed files and use your deployment platform's secret manager in non-local environments.
