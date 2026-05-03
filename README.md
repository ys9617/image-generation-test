# Minimal Image Generation Site

A small React + Express project for internal image generation and shareable prompt workflows.

## Features

- ChatGPT-style prompt UI
- Image generation via OpenAI Images API
- Shareable prompt URLs
- Local backend for API key security

## Setup

1. Install dependencies:
   ```bash
   npm install
   ```

2. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="your_api_key"
   ```

3. Run locally:
   ```bash
   npm run dev
   ```

4. Open the frontend:
   - http://localhost:5173

## Deployment

Deploy both frontend and backend together on platforms like Vercel, Railway, or Render.
Set `OPENAI_API_KEY` in production environment variables.

## Notes

- The frontend proxies `/api` to `http://localhost:4000` in development.
- Generated prompt URLs can be shared and will prefill the input on load.
