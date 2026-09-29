# GPT Associate

A live meeting and interview assistant. It listens to the conversation, notices when you are asked something, and shows a ready-to-say answer in the first person within about 3 seconds. It builds on background you upload before the meeting and on everything said during it. Answers are text only, on a web page you can open on the meeting computer and on your phone.

- **Listens continuously.** The other side (a Google Meet or Teams tab, system audio, or a loopback device) and your mic are transcribed separately, labelled *Them* and *Me*. In-person mode uses one mic, labelled *Room*. Listening never pauses while an answer is being written.
- **Answers automatically or on demand.** Questions from the other side are detected instantly and at no cost. You can also press **Answer now** (Space) or type a question.
- **Prepares from your background.** Upload a job description, resume, prep notes and transcripts of past interviews (PDF, DOCX, TXT, MD) into a saved profile. Each file gets a type, and everything is turned once into a *prep pack*: what they want, how you match, a story bank, a question bank (questions asked before, your prepared ones and likely ones, each with a ready answer) and lessons from past rounds. Live answers build on the matching prepared answer.
- **Reuses what you have.** A job profile can build on a base profile (for example "My career" with your resume and stories). Any meeting recorded in the app can be added to a profile as a past interview, or downloaded as a Markdown summary with a Chinese digest.
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

1. Optional: create a base profile such as "My career" with your resume, stories and general prep notes.
2. Open **Background**, create a profile for the meeting (for example "Acme interview"), set **Builds on** to the base profile, and fill in *About me*, *Meeting goal / job description*, *About the other side*, and *Keywords* (names and jargon that help transcription).
3. Upload files and check each file's type:

   | Type | Use it for | How it is used |
   |---|---|---|
   | Job description / meeting goal | the JD, the meeting agenda | requirements the answers are tied to |
   | Resume / about me | resume, LinkedIn export | facts for answers and stories |
   | My prep notes / prepared answers | your Q&A notes, STAR stories (English or Chinese) | kept as your prepared answers |
   | Company / interviewer info | research on the company and interviewers | context for answers and questions to ask |
   | Past interview / meeting transcript | recordings from Otter, Feishu, Zoom or this app | the questions they asked, how you answered, where you were weak. Searched live, never pasted verbatim |

4. Press **Save & prepare**. The prep pack is written once (up to a minute) and rewritten only when the material changes. Read it before the meeting and edit it if you like. The status line shows what is used, for example "prep pack + full text + search over 2 docs".
5. Profiles are saved under `data/` and are reused next time.

## During the meeting

Pick the meeting type and press **Start listening**:

| Situation | Mode | "Them" source |
|---|---|---|
| Google Meet or Teams in the browser | Online call: them + my mic | *Share tab / screen audio*: pick the meeting tab and turn on **Share tab audio** |
| Teams or Zoom desktop app on Windows | Online call: them + my mic | *Share tab / screen audio*: pick **Entire screen** and turn on **Share system audio** |
| Teams desktop app on macOS | Online call: them + my mic | macOS 14.2+ with a recent Chrome: *Entire screen* + **Share system audio**. Otherwise install [BlackHole](https://existential.audio/blackhole/), set up a Multi-Output Device (speakers + BlackHole) as the Mac's output, and choose *Them: BlackHole* in the source list |
| In a room with people | In person: mic only | none |

- **Auto-answer** answers questions from the other side as soon as they finish speaking. If they keep talking right after the question, the answer restarts with the full question.
- **Answer now** (Space) answers what was just said, even mid-sentence.
- **Answer in details** (D) on an answer card writes a fuller answer (about 120 to 200 words) with a stronger model that thinks briefly first. Typed questions (Enter in the box at the bottom) are always answered this way. Thinking is capped at 5 seconds: if no words have appeared by then, the fast model writes the detailed answer instead.
- **Improve** (I) compares your spoken reply with your background and suggests what to add.
- **Search web** on an answer card re-answers it using live internet results.
- **Live notes** shows what the assistant has learned. You can add facts by hand.
- **Summary** downloads this meeting as Markdown: a Chinese digest (中文要点), the questions they asked and how you answered, facts you stated, follow-ups, then the full transcript.
- **New meeting** saves the transcript and answers to `data/sessions/` and starts fresh.

## After the meeting

In **Background → Past meetings**, pick a meeting and press **Add to background** to add its summary and transcript to the current profile as a past interview, then **Save & prepare**. The next round's prep pack then covers what they asked, where you were weak and what you already told them. **Download summary** saves any past meeting as `.md`.

Use headphones for online calls. Otherwise your mic also picks up the other side from the speakers. An echo filter drops most of that, but headphones are cleaner.

## Speed and cost

The default models are the fastest and cheapest as of September 2026, and all of them can be changed in `.env`.

| Job | Default | Price |
|---|---|---|
| Speech to text | `gpt-4o-mini-transcribe` over the Realtime API, server VAD | $0.003 per audio minute, per stream |
| Answers, notes, prep pack, summary | `gpt-6-luna`, reasoning effort `none` (prep pack and summary: `low`), answers streamed | $0.10 / $0.01 cached / $0.50 per 1M tokens |
| Detailed answers (button and typed questions) | `gpt-6-sol`, reasoning effort `low`, thinking capped at 5 s | $2.00 / $0.20 cached / $10.00 per 1M tokens, about 1 to 3 cents per answer |
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
| `app/meeting.py` | Turn tracking, auto-answer, answer streaming, Improve, web search, live notes, echo guard, prep pack, meeting summaries |
| `app/answerer.py` | Prompts and Responses API calls, laid out static-first for prompt caching |
| `app/store.py` | Profiles, typed documents, base profiles, saved meetings and their summaries under `data/` |
| `app/knowledge.py` | Document types, extraction, the static background block (prep pack + full text), BM25 retrieval (English and Chinese), transcription keywords |
| `app/detector.py` | Question and echo heuristics |
| `app/server.py` | FastAPI routes, access code, `/ws/ui` and `/ws/audio` |
| `static/` | The page, and `pcm-worklet.js`, which turns captured audio into 24 kHz PCM16 |

Transcripts and uploaded material stay on your computer under `data/`. Audio and text are sent only to OpenAI, with `store=false` on every Responses call. Check the rules on recording and on assistive tools for your meeting and location.
