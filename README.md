# GPT Associate

A live meeting and interview assistant. It listens to the conversation, notices when you are asked something, and shows a ready-to-say answer in the first person within about 3 seconds. It builds on background you upload before the meeting and on everything said during it. Answers are text only, on a web page you can open on the meeting computer and on your phone.

- **Listens continuously.** The other side (a Google Meet or Teams tab, system audio, or a loopback device) and your mic are transcribed separately, labelled *Them* and *Me*. In-person mode uses one mic, labelled *Room*. Listening never pauses while an answer is being written.
- **Answers automatically or on demand.** Questions from the other side are detected instantly and at no cost. You can also press **Answer now** (Space) or type a question.
- **Knows your background.** Upload a resume, job description or notes (PDF, DOCX, TXT, MD) into a saved profile. Small material goes into every prompt in full. Large material is condensed once into a brief, plus keyword search.
- **Learns during the meeting.** Running notes about who is who, what they care about and what you have already said are refreshed in the background every ~90 s.
- **Improve** (I) checks what you actually said against your background and gives you a line to add now.
- **Search web** re-answers a question with live internet results. It is on demand only, because it takes 3–6 s.

## Setup

Requires Python 3.11+ and an OpenAI API key. The meeting computer should use Chrome or Edge, which can capture tab audio.

```bash
git clone https://github.com/SmallDataMatrix/GPT-Associate.git
cd GPT-Associate
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # Windows: copy .env.example .env
# edit .env and set OPENAI_API_KEY
python -m app
```

The terminal prints two things:
- **`http://localhost:8000`** for the meeting computer.
- **A phone URL** with a 6-digit access code and a QR code. Open it on a phone or tablet on the same Wi-Fi to read answers on a second screen.

Run the app **on the computer that is in the meeting.** Browsers only allow audio capture on `localhost` (or HTTPS), so the phone page is for reading and triggering answers, not for capturing.

## Before the meeting

1. Open **Background**, create a profile (for example "Acme interview") and fill in *About me*, *Meeting goal / job description*, *About the other side*, and *Keywords* (names and jargon that help transcription).
2. Upload files, then press **Save & prepare**. The status line tells you whether the full text or a brief is used. You can edit a generated brief.
3. Profiles are saved under `data/` and are reused next time.

## During the meeting

Pick the meeting type and press **Start listening**:

| Situation | Mode | "Them" source |
|---|---|---|
| Google Meet or Teams in the browser | Online call: them + my mic | *Share tab / screen audio*: pick the meeting tab and turn on **Share tab audio** |
| Teams or Zoom desktop app on Windows | Online call: them + my mic | *Share tab / screen audio*: pick **Entire screen** and turn on **Share system audio** |
| Teams desktop app on macOS | Online call: them + my mic | macOS 14.2+ with a recent Chrome: *Entire screen* + **Share system audio**. Otherwise install [BlackHole](https://existential.audio/blackhole/), set up a Multi-Output Device (speakers + BlackHole) as the Mac's output, and choose *Them: BlackHole* in the source list |
| In a room with people | In person: mic only | none |

- **Auto-answer** answers questions from the other side as soon as they finish speaking. If they keep talking right after the question, the answer restarts with the full question.
- **Answer now** (Space) answers what was just said, even mid-sentence. You can also type a question and press Enter.
- **Improve** (I) compares your spoken reply with your background and suggests what to add.
- **Search web** on an answer card re-answers it using live internet results.
- **Live notes** shows what the assistant has learned. You can add facts by hand.
- **New meeting** saves the transcript and answers to `data/sessions/` and starts fresh.

Use headphones for online calls. Otherwise your mic also picks up the other side from the speakers. An echo filter drops most of that, but headphones are cleaner.

## Speed and cost

The default models are the fastest and cheapest as of September 2026, and all of them can be changed in `.env`.

| Job | Default | Price |
|---|---|---|
| Speech to text | `gpt-4o-mini-transcribe` over the Realtime API, server VAD | $0.003 per audio minute, per stream |
| Answers, notes, brief | `gpt-6-luna`, reasoning effort `none`, streamed | $0.10 / $0.01 cached / $0.50 per 1M tokens |
| Web search (button only) | Responses API `web_search` | about $0.01 per search |

**A one-hour online interview costs about $0.40,** almost all of it transcription. Switch to *them only* to halve that.

How answers stay fast and cheap:
- The prompt starts with the fixed instructions and background, marked as cache breakpoints, so repeated input is served from OpenAI's prompt cache.
- Question detection is local and free.
- The cache and connection are warmed when you start.
- Answers are short and stream as they are written.
- Every answer card shows "first words" and "complete" times, measured from when the speaker stopped.

The header shows the running cost and the cache hit rate.

## Development

```bash
pip install -r requirements.txt
python -m pytest
```

| Path | Role |
|---|---|
| `app/transcriber.py` | Relays each audio source to an OpenAI Realtime transcription session; reconnects automatically |
| `app/meeting.py` | Turn tracking, auto-answer, answer streaming, Improve, web search, live notes, echo guard |
| `app/answerer.py` | Prompts and Responses API calls, laid out static-first for prompt caching |
| `app/knowledge.py` | Document extraction, the full-text or brief decision, BM25 retrieval, transcription keywords |
| `app/detector.py` | Question and echo heuristics |
| `app/server.py` | FastAPI routes, access code, `/ws/ui` and `/ws/audio` |
| `static/` | The page, and `pcm-worklet.js`, which turns captured audio into 24 kHz PCM16 |

Transcripts and uploaded material stay on your computer under `data/`. Audio and text are sent only to OpenAI, with `store=false` on every Responses call. Check the rules on recording and on assistive tools for your meeting and location.
