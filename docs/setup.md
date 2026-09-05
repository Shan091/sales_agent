# Setup

A cold start, from `git clone` to a paid order confirmed in chat. Roughly 30 minutes, most of it
waiting on Meta and on `pip install torch`.

- **What the pieces are:** [architecture.md](architecture.md)

## What you need

| | why |
|---|---|
| Docker + Docker Compose | Postgres/pgvector and Redis; the app runs here too |
| Python 3.11+ | only if you want to run the app on the host instead of in a container |
| An OpenRouter key (or any OpenAI-compatible endpoint) | the LLM calls |
| A Meta developer app with WhatsApp product added | inbound and outbound messages |
| A Razorpay account in **test mode** | payment links. Optional — the agent degrades to no checkout |
| A tunnel (`ngrok`, `cloudflared`) | Meta and Razorpay must reach your machine over HTTPS |

~10GB of disk and RAM headroom: `requirements.txt` pulls `torch`, and the `bge-m3` embedding model is
~2.2GB.

---

## 1 · Clone and configure

```bash
git clone https://github.com/Shan091/sales_agent.git
cd sales_agent
cp .env.example .env
```

Open `.env` and set these five before anything else works:

```ini
OPENAI_API_KEY=sk-or-...
META_APP_SECRET=...
WHATSAPP_API_TOKEN=...
WHATSAPP_PHONE_NUMBER_ID=...
WHATSAPP_VERIFY_TOKEN=any-string-you-invent
```

Everything else in `.env.example` has a working default and is commented with what it costs you to
change it. Two worth reading before you start: **`DEFAULT_MODEL` and `FAST_MODEL` must be paid
endpoints** — a `:free` OpenRouter variant is pooled and queued, and produced 15–42s turns here — and
`APP_ENV=production` makes startup hard-fail on any placeholder secret, which is what you want on a
real deployment and not on your laptop.

---

## 2 · Bring the stack up

```bash
docker compose up -d --build
docker compose ps          # postgres, redis, api, worker — all healthy
curl localhost:8000/health
```

Four services: `postgres` (pgvector/pg16), `redis`, `api` (uvicorn on :8000), `worker` (TaskIQ). The
`api` container runs `python -m src.scripts.init_db` on start, which creates the `vector` extension,
the tables and the HNSW/GIN indexes.

**`docker-compose.yml` overrides `DATABASE_URL` and `REDIS_URL`** with in-network hostnames, so the
values in `.env` only matter if you run on the host. The Postgres container is created with
`POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` from `.env` (defaulting to `otohom` for all
three), and the overridden DSN is built from the same values — change one and both follow.

> **The worker is pinned to `--workers 1` on purpose.** TaskIQ's default spawned two processes that
> each loaded their own ~2GB copy of the embedding model, for no throughput gain: a conversation is
> serialised by the Redis mutex and everything else is I/O-bound async.

### Or on the host

```bash
pip install -r requirements.txt        # torch + sentence-transformers; expect a few minutes
python -m src.scripts.init_db
uvicorn src.main:app --reload          # terminal 1
taskiq worker src.tasks.broker:broker  # terminal 2
```

Run everything **from the repo root** — imports are absolute (`src.*`, `config.*`) and there is no
packaging shim. You still need Postgres and Redis; `docker compose up -d postgres redis` is the easy
way. Point `DATABASE_URL` at `localhost:5432` with the compose credentials, not the `.env.example`
placeholder — that placeholder user is why host-side `init_db` / `seed_pricing` / `ingest_catalog`
fail authentication while the containers work fine.

---

## 3 · Seed prices, ingest the catalogue

**Prices are required before checkout can quote anything.** With an empty catalogue
`_build_pending_order` returns `None` and the agent produces no quote at all.

```bash
docker compose exec -T api python -m src.scripts.seed_pricing
```

Idempotent upsert of 34 products into region `IN-KL`. These are **representative test-mode figures,
labelled as such in the script — not official Otohom retail pricing.**

Then the RAG corpus, one category folder at a time. Run these **inside the container**:

```bash
for c in switches security sensors curtains hubs; do
  docker compose exec -T worker python -m src.scripts.ingest_catalog \
    --input-dir ./docs/catalog/$c --doc-type TECHNICAL_SPEC --category $c
done

docker compose exec -T worker python -m src.scripts.ingest_catalog \
  --input-dir ./docs/catalog/company --doc-type PRODUCT_CATALOG --category company
```

Expect **77 chunks** across the six folders (22 parents, 55 children). Ingest is idempotent by
`source_hash`: re-running changes nothing, and editing a file re-chunks only that document.

> If you edit `docs/catalog/**`, the change reaches the agent only after a re-ingest. pgvector holds
> the copy the agent reads, not the markdown.

### Talk to it without WhatsApp

```bash
docker compose exec worker python -m src.scripts.local_chat
```

A terminal REPL against the real graph, real retrieval and real pricing — no Meta, no tunnel, no
Redis. This is the fastest loop for prompt and pricing work; multi-turn memory is in-process, so one
run is one conversation. For the *real* WhatsApp thread,
`python -m src.scripts.reset_thread 919812345678` clears its checkpoints and dedup flags for a clean
demo re-run (audit rows are left alone unless you pass `--purge-records`).

---

## 4 · Expose the app

Meta and Razorpay both push to you, and both require HTTPS.

```bash
ngrok http 8000
```

Put the URL in `.env` — **without a trailing slash** — and restart the api container:

```ini
PUBLIC_BASE_URL=https://abc123.ngrok-free.app
```

That one value is all the brochure needs: `settings.brochure_url` derives
`https://…/api/v1/brochure` from it, so there is no second URL to keep in sync. It is only used if
`BROCHURE_FILE_PATH` actually resolves to a file — **the merchant's artwork is not in this
repository**, so on a fresh clone there is nothing to serve and the agent is told it has no brochure
and must not offer one. Drop your own PDF at `docs/brochure.pdf`, or set `BROCHURE_URL` to a CDN copy.

> A Drive or Dropbox "view" link will not work. Meta's servers fetch the URL themselves and need a
> direct `application/pdf`; a share link returns HTML and is rejected.

A free ngrok URL changes every restart. Each time it does, update `PUBLIC_BASE_URL` **and** the
callback URLs in the Meta and Razorpay dashboards.

---

## 5 · Meta / WhatsApp Cloud API

1. **developers.facebook.com** → your app → **WhatsApp → Configuration**.
2. **Callback URL:** `https://<your-tunnel>/api/v1/webhooks`
3. **Verify token:** the same string you invented for `WHATSAPP_VERIFY_TOKEN`.
4. Click **Verify and save.** The GET handler answers the challenge; if this fails, the token doesn't
   match or the api container isn't reachable.
5. **Webhook fields:** subscribe to **`messages`**. Nothing else is needed.
6. **App secret:** *Settings → Basic → App secret* → `META_APP_SECRET`. Every inbound POST is
   HMAC-verified over the exact raw body, so a wrong secret means every message 401s. (The *verify*
   handshake in step 4 is a different check and answers 403 when its token is wrong.)
7. **Test number:** *API Setup* gives you a test sender and a 24-hour token. Add your own number as a
   recipient there before you can message it.

**Unverified apps are capped at 250 conversations/day** and can only message numbers you've added as
recipients. That is enough to demo the whole flow; a production rollout needs Meta business
verification, which is a merchant-side process, not a code one.

---

## 6 · Razorpay (test mode)

Optional. Leave the keys empty and the agent runs with checkout disabled — it sells, quotes and hands
off, but never mints a link. `settings.razorpay_enabled` gates every money action on the key pair.

1. **Dashboard → Settings → API Keys → Generate Test Key.** Copy both halves:
   ```ini
   RAZORPAY_KEY_ID=rzp_test_...
   RAZORPAY_KEY_SECRET=...
   ```
   Make sure the dashboard's **Test Mode** toggle is on. A live key here charges real money.
2. **Dashboard → Settings → Webhooks → Add New Webhook.**
   - **URL:** `https://<your-tunnel>/api/v1/webhooks/razorpay`
   - **Secret:** invent one, and put the same string in `RAZORPAY_WEBHOOK_SECRET`.
   - **Active events:** `payment_link.paid`, `payment_link.expired`, `payment.failed`,
     `payment.captured`.
3. Restart both containers so the new values load.

> **Razorpay sends more than one event for the same money.** `payment_link.paid` and
> `payment.captured` arrive with different event ids, so the event-id dedupe cannot collapse them.
> That is expected and handled: the confirmation sequence is claimed as a whole, so the second event
> is silent. Subscribe to both — `payment.captured` is the one that carries the payment id.

**Empty `RAZORPAY_WEBHOOK_SECRET` means inbound webhooks are rejected**, deliberately: an unverifiable
"paid" event is worse than no checkout. And with `APP_ENV=production`, a *half*-configured setup
(keys without a webhook secret, or the reverse) hard-fails at startup for the same reason.

### Test cards and UPI

| instrument | value | outcome |
|---|---|---|
| Visa | `4111 1111 1111 1111` | success |
| Mastercard | `5267 3181 8797 5449` | success |
| Visa | `4000 0000 0000 0002` | **declined** — use this to walk the 1-2-3 decline ladder |
| UPI | `success@razorpay` | success |
| UPI | `failure@razorpay` | failure |

Any future expiry, any CVV, any name. The declined card is worth using at least once: the first
decline resends the live link with **no** human button, the second adds the hatch, the third
(`MAX_PAYMENT_FAILURES`) escalates to a person.

### Money bounds

Set once in `.env` and clamped in code, not in a prompt:

```ini
MAX_DISCOUNT_PCT=12.0        # ceiling on any agent-applied discount
RAZORPAY_MIN_AMOUNT=1.0
RAZORPAY_MAX_AMOUNT=500000.0 # per-order cap; above this no link is minted
MAX_LINE_QTY=20
```

---

## 7 · Lead delivery

Two independent sinks; each fires only if configured, so a partial setup still works. Both carry the
same row, and the row's **`kind`** column says what job it is:

| `kind` | means |
|---|---|
| `LEAD` | somebody to call back about a product |
| `ESCALATION` | somebody waiting for a person, right now |
| `SUPPORT` | an existing customer with a problem |
| `PARTIAL` | a thread with no product interest — not a callable lead |

One destination is a deliberate choice, not a limitation: two tabs is two places to forget to look, so
the team filters on `kind`. Dedup is **per kind**, so a hot lead that later needs a person appears as
both, exactly once each.

### Sink 1 — webhook (the flexible one)

```ini
LEADS_WEBHOOK_URL=https://script.google.com/macros/s/AKfy.../exec
```

POSTs the row as JSON. Point it at Google Apps Script, Zapier, Make or n8n and fan out from there —
**the WhatsApp Sales Leads group has to go through this**, because the Meta Cloud API cannot post to
groups directly.

A Google Sheet in four steps: create the sheet → **Extensions → Apps Script** → paste this → **Deploy
→ New deployment → Web app**, execute as *me*, access *anyone*, and copy the `/exec` URL.

```javascript
function doPost(e) {
  const d = JSON.parse(e.postData.contents);
  const sheet = SpreadsheetApp.getActiveSpreadsheet().getSheets()[0];
  sheet.appendRow([
    new Date(), d.kind, d.route, d.mobile_number, d.customer_name, d.city,
    d.products_interested, (d.pain_points || []).join("; "), d.stage, d.order,
    d.escalation_reason, d.archetype, d.language, d.summary
  ]);
  return ContentService.createTextOutput(JSON.stringify({ok: true}))
    .setMimeType(ContentService.MimeType.JSON);
}
```

Matching header row, left to right:

```
Timestamp | Kind | Route | Mobile | Name | City | Products | Pain points | Stage | Order | Escalation reason | Archetype | Language | Summary
```

**Redeploy after any edit** — Apps Script serves the last *deployed* version, not the last saved one,
which is the single most common reason a working script stops appending rows.

### Sink 2 — email

Independent of the webhook. Leave `SMTP_HOST` empty to disable it.

```ini
LEADS_EMAIL_TO=sales@otohom.com
LEADS_EMAIL_FROM=agent@example.com
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=agent@example.com
SMTP_PASSWORD=your-app-password
SMTP_USE_TLS=true
```

Gmail needs an **app password** (2FA on → *Google Account → Security → App passwords*), not your
account password.

`LeadSink.build_email` writes **one format for every kind**, meant to be read on a phone: subject, the
facts in a fixed order, the digest indented under its own heading, and — for anything waiting on a
person — the exact `#done` command to hand the thread back. An alert that doesn't say what to do next
gets read and forgotten.

Delivery is **fail-soft and never raises**: a downstream outage cannot break the customer's turn or
trigger a task retry that would re-send WhatsApp replies. Check the worker log for
`LEAD delivery result:` / `ESCALATION delivery result:` to see what actually went out.

---

## 8 · Staff handoff numbers

Who may take a thread back from the agent, comma-separated. Any spelling works — matching is on the
**last ten digits**, because colleagues type numbers the way they say them while Meta delivers the
full international form.

```ini
STAFF_WHATSAPP_NUMBERS=9812345678,919887654321
HANDOFF_MAX_HOLD_HOURS=24
```

**This list is the authorisation.** A number not on it is treated as an ordinary customer and gets an
ordinary sales reply — deliberately not an error and not a hint, because anyone who learned the syntax
could otherwise release any hold on any thread. Empty (the default) disables the commands entirely.

A colleague works the customer on their **own** number, so nothing they say reaches this system.
Texting the agent's number is how the hold is released and how the outcome gets into the agent's
context:

| text this to the agent's number | effect |
|---|---|
| `#done 919812345678 Called, quoted 4 locks, sending invoice Monday` | releases the hold silently |
| `#back 919812345678 All sorted, they'll message you` | releases **and** tells the customer the agent is back |
| `#status 919812345678` | reports the hold, changes nothing |

There is a CLI equivalent if you have a terminal —
`python -m src.scripts.resolve_handoff 919812345678 --note "..." [--tell-customer]` — but the WhatsApp
commands exist because a salesperson has a phone, and holds were otherwise running all the way to the
`HANDOFF_MAX_HOLD_HOURS` safety net with the outcome never recorded.

---

## 9 · The end-to-end rehearsal

Walk this once before you trust it. Each step names what to check, not just what to send.

| # | do this | expect |
|---|---|---|
| 0 | `docker compose ps` | four services, all healthy |
| 1 | `curl localhost:8000/health` | `200` |
| 2 | WhatsApp *"hi"* to the test number | typing dots, then a greeting that offers themes (bills, safety, curtains, convenience) — **not** a product menu |
| 3 | *"I want a smart lock for my front door"* | a discovery question, no price yet |
| 4 | *"what's the battery life in months?"* | a deterministic "let me get you exact details" — **this is correct.** No Otohom document states it, so the agent refuses rather than inventing a number |
| 5 | *"the base one"* | the **step-up card**: benefit heading, bullets, `_(+₹…)_` inside the product line, and *"You'd get this one instead of the one you picked."* Buttons: `Switch to Premium` / `Keep the Base` / `Explore more` |
| 6 | tap `Keep the Base` | the **pairing card** — a different product, with a reachable reward named |
| 7 | tap the keep-going button | *"Shall I show you the price?"* and a hook button. **Two buttons only** |
| 8 | tap the **other** button (`Explore more`) | a browse answer that does **not** show the total, with `Yes, show the price` still in slot one. This is the `hold` beat |
| 9 | tap `Yes, show the price` | the itemised order: per-line all-in figures with the fitting split, the Total, *"Just type it in the message box"*, and `Apply N% off`-or-`Add ‹X›` / `Confirm & pay` / `Explore more` |
| 10 | type *"make it two"* | a re-priced order, `checkout_confirmed` cleared, offer tier re-selected |
| 11 | tap `Confirm & pay` | **no link yet** — *"what name and city should I put on the order?"* |
| 12 | reply *"Anil, Kochi"* | the Razorpay link, on the same authorisation |
| 13 | pay with `4000 0000 0000 0002` | the live link again, **no human button** |
| 14 | pay with `4111 1111 1111 1111` | celebration, then the receipt — **in that order** |
| 15 | `select * from payment_orders order by created_at desc limit 1;` | one row: order id, amount, status `paid`, the payment id, `audit_notes` |
| 16 | check the sheet and the inbox | one `LEAD` row, one email, the Summary column populated |
| 17 | *"I want to speak to someone"*, twice | a handoff line matching the reason; then an ordinary message gets a short **holding** line, not a sales turn |
| 18 | from a staff number: `#done <customer> tested` | silent release; the next customer message routes normally and the agent knows the note |

Step 4 is the one people mistake for a bug. Four things customers ask for are in **no** Otohom
document — door-lock battery life in months, RFID, an anti-tamper alarm, and the 6 SW FAN's fan
wattage. The brochure gives chemistry and capacity, says "card swiping" without naming the technology,
says "anti-pry alarm", and gives the fan channel as `16A(1)Fan`. Filling those in from a plausible
number is not an option: someone fits these to their front door.

### Tests

635 tests, of which **632 need no infrastructure at all** — no Postgres, no Redis, no API key, no model
download:

```bash
pip install -r requirements-dev.txt      # requirements.txt + pytest and pytest-asyncio
pytest --deselect tests/test_rag_pipeline.py::TestHybridSearch \
       --deselect tests/test_rag_pipeline.py::TestTheLiveCatalogueAnswersRealQuestions
# 632 passed, 3 deselected

pytest tests/test_payments.py -v         # the money guarantees: clamp, guardrail, signature, gates
pytest tests/test_autonomy.py -v         # autonomy both ways + the critical escalation valve
```

The test runner is deliberately **not** in `requirements.txt` — the Docker images build from that file, and
a production worker image should not ship a test runner. So the suite runs on the host, not inside the
container.

The three deselected ones are the two retrieval gates, and they are deselected rather than skipped on
purpose: they need live Postgres + pgvector **and** the 2.2 GB embedding model, and a test that quietly
skips itself is a test nobody notices has stopped running. `TestHybridSearch` asks *does the pipeline
work* against its own fixture corpus (10/10); `TestTheLiveCatalogueAnswersRealQuestions` asks *can a
customer get an answer* against the **ingested** corpus, writing and deleting nothing (12/12). Run them
from the host, after an ingest, with the compose stack up — `DATABASE_URL` must point at the same
Postgres the worker uses, or they fail on authentication rather than on retrieval:

```bash
pytest tests/test_rag_pipeline.py -v
```

---

## 10 · Troubleshooting

| symptom | cause | fix |
|---|---|---|
| Meta's **Verify and save** fails | `WHATSAPP_VERIFY_TOKEN` mismatch, or the tunnel isn't up | compare the strings character-for-character; `curl` the callback URL yourself |
| every inbound message 401s | wrong `META_APP_SECRET` | *Settings → Basic → App secret*. The HMAC is over the exact raw body and is fail-closed by design |
| webhook returns 200, nothing happens | the worker isn't running, or Redis is down | `docker compose logs -f worker` |
| a message is silently ignored | Redis `SETNX` already saw that `message.id` | expected on a Meta retry. `reset_thread` if you're re-running a demo |
| **no quote at all**, ever | prices aren't seeded, or the model invented a sku | run `seed_pricing`; check the worker log for a sku that didn't resolve |
| the agent says it can't find specs | retrieval didn't match, or the fact genuinely isn't in the corpus | did you ingest? Is the product named in `docs/catalog/**`? If it isn't, the refusal is correct |
| retrieval returns nothing at all | corpus not ingested, or the embedding model didn't load | count rows: `select count(*) from document_chunks;` — expect 77 |
| turns take 15–40s | `DEFAULT_MODEL` / `FAST_MODEL` point at a `:free` endpoint | use paid model ids. This was the whole of the latency problem |
| the first message of the day is slow | you restarted the worker; the boot warm-up is still running | wait for `[warm]` in the log before the demo |
| Razorpay webhook 401s | `RAZORPAY_WEBHOOK_SECRET` mismatch or empty | it's fail-closed on purpose. Copy the dashboard secret exactly. A 400 instead means the body wasn't JSON |
| paid, but no confirmation in chat | webhook can't reach you, or events aren't subscribed | Razorpay's dashboard shows delivery attempts and the response body |
| paid, but `payment_orders` is empty | old database, pre-`timestamptz` | `python -m src.scripts.init_db` — it ALTERs the columns idempotently |
| host-side scripts fail authentication | `.env`'s `DATABASE_URL` is the placeholder user; the containers use the compose override | pass the compose credentials, or run the script inside the container |
| `docker compose up` OOM-kills the worker | two model copies, or a low Docker memory limit | the compose file pins `--workers 1`; raise Docker Desktop's memory to ~8GB |
| a voice note gets a "please type it" reply | transcription is not implemented | by design. A placeholder transcript is indistinguishable downstream from something the customer actually said |
| typing dots appear *after* the reply | you're looking at an instant no-LLM answer | expected and handled — the heartbeat suppresses dots when the reply already started |
