# Daily News Briefing Bot

Fetches today's news via NewsAPI + RSS feeds, summarises each category with Gemini 2.5 Flash, and emails a clean HTML briefing via Resend.

---

## Quick start

```
cd news-briefing
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env   # then edit .env with your keys
python main.py --dry-run
```

---

## API Keys

### 1. Google Gemini

1. Go to **https://aistudio.google.com/app/apikey**
2. Click **Create API Key**
3. Copy the key into `.env` as `GEMINI_API_KEY`

The script uses `gemini-2.5-flash`. The free tier (Google AI Studio) covers daily use comfortably (~4 calls/day).

---

### 2. NewsAPI

1. Go to **https://newsapi.org/register** and sign up for a free account
2. Your API key is shown immediately on the dashboard
3. Copy it into `.env` as `NEWSAPI_KEY`

Free tier: 100 requests/day, articles up to 30 days old. More than enough.

---

### 3. Resend (email)

Resend requires a **verified sending domain** to deliver to external addresses.

1. Sign up at **https://resend.com** (free tier: 100 emails/day, 3,000/month)
2. Go to **Domains → Add Domain** and add a domain you own (e.g. `yourdomain.com`)
3. Add the DNS records Resend shows you (takes 1–5 minutes to verify)
4. Go to **API Keys → Create API Key** — copy the key (`re_...`) into `.env` as `RESEND_API_KEY`
5. Set `FROM_EMAIL` to an address on your verified domain, e.g. `briefing@yourdomain.com`

> **No domain?** Resend gives every account a shared test address `onboarding@resend.dev`.
> You can set `FROM_EMAIL=onboarding@resend.dev` and add your own Gmail as a
> verified recipient under **Resend → Audiences** for free-tier testing.
> To send to *any* external address you must own a verified domain.

---

## Testing

```bash
# Print briefing to console — no email sent, no Resend key needed
python main.py --dry-run
```

Check the output looks right, then run without the flag to send the real email.

---

## Schedule on Windows (Task Scheduler)

1. **Open Task Scheduler** → *Create Basic Task*
2. **Name**: "Morning News Briefing"
3. **Trigger**: Daily, pick your time (e.g. 07:00)
4. **Action**: Start a program
   - Program: `C:\Users\USER\news-briefing\venv\Scripts\python.exe`
   - Arguments: `main.py`
   - Start in: `C:\Users\USER\news-briefing`
5. Finish and **Run** to test immediately

Or run via a one-line batch file (`run.bat`) if you prefer:

```bat
@echo off
cd /d C:\Users\USER\news-briefing
venv\Scripts\python.exe main.py >> logs\briefing.log 2>&1
```

Then point Task Scheduler at `run.bat` to get persistent logs.

---

## Customisation

| What | Where |
|---|---|
| Add/remove RSS feeds | `FEEDS` dict in `main.py` |
| Change recipient email | `TO_EMAIL` in `.env` |
| Adjust bullet count / tone | `PROMPTS` dict in `main.py` |
| Change Gemini model | `MODEL` constant in `main.py` |
| NewsAPI search terms | `NEWSAPI_QUERIES` dict in `main.py` |
