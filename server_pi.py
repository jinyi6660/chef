from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context
from flask_cors import CORS
import socket
import anthropic
from openai import OpenAI
import requests
import base64
import json
import os
import re
import time
import threading

app = Flask(__name__)
CORS(app)

display_state = {
    "status": "idle",
    "ts": 0,
    "answers": {},
    "emotionTags": [],
    "imagePrompt": "",
    "dishName": "",
    "dishDesc": "",
    "image": None,
    "ingredientImages": [],
    "relayLines": [],
    # showPhase drives the handshake between the two screens: the kitchen
    # display (display_test.html) advances it as its own performance plays
    # out, and the ordering screen (index.html) waits for "done" here
    # instead of the raw backend `status` before showing the receipt.
    "showPhase": "idle",   # idle | greeting | pageturn | reading | cooking | done
    "greetingAudio": None,
    # bumped by /reset — kitchen.html, order.html, and receipt.html all
    # watch this and hard-reload themselves when it changes, so clicking
    # Finish gives every screen a genuinely clean slate (no leftover JS
    # state/stuck audio/timers) for the next guest instead of trying to
    # softly reset each screen's own in-memory state.
    "resetSignal": 0,
}

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Reference images fed to gpt-image-2's edit endpoint alongside the text
# prompt for the final dish/cake image, so it has real visual anchors for
# the numeral candles, the eerie black-background mood, and the base
# birthday-cake shape, instead of guessing purely from text.
REFERENCE_IMAGES = [
    os.path.join(BASE_DIR, "ref1 (18).jpg"),  # silver numeral candles — digit shape reference
    os.path.join(BASE_DIR, "ref1 (14).jpg"),  # eerie black-bg dramatic cake — mood/lighting reference
    os.path.join(BASE_DIR, "ref1 (2).jpg"),   # classic birthday cake — base shape/candle placement reference
]

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "C5rrTayXl4m3QBraxgkY"  # cloned voice "jinyi"

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
openai_client = OpenAI(api_key=OPENAI_API_KEY)


@app.route("/state", methods=["GET"])
def get_display_state():
    state = {k: v for k, v in display_state.items() if k not in ("image", "greetingAudio")}
    state["hasImage"] = bool(display_state.get("image"))
    state["hasGreetingAudio"] = bool(display_state.get("greetingAudio"))
    return jsonify(state)

@app.route("/state/image", methods=["GET"])
def get_display_image():
    return jsonify({"image": display_state.get("image")})

@app.route("/state/voice", methods=["GET"])
def get_greeting_audio():
    return jsonify({"audio": display_state.get("greetingAudio")})

@app.route("/reset", methods=["GET", "POST"])
def reset_state():
    """Manually resets display_state back to idle, for testing without
    restarting the server — otherwise a leftover 'done'/'cooking' status
    from the last order makes kitchen.html jump straight into the show
    on refresh instead of sitting on the standby frame."""
    display_state.update({
        "status": "idle",
        "ts": 0,
        "answers": {},
        "emotionTags": [],
        "imagePrompt": "",
        "dishName": "",
        "dishDesc": "",
        "image": None,
        "ingredientImages": [],
        "relayLines": [],
        "showPhase": "idle",
        "greetingAudio": None,
        "resetSignal": display_state.get("resetSignal", 0) + 1,
    })
    return jsonify({"status": "reset"})

@app.route("/phase", methods=["POST"])
def set_phase():
    """The kitchen display calls this as its own performance advances
    (greeting -> pageturn -> reading -> cooking -> done). The ordering
    screen waits for showPhase == 'done' before it reveals the receipt,
    instead of the raw backend status."""
    data = request.json or {}
    phase = data.get("phase")
    if phase:
        display_state["showPhase"] = phase
    return jsonify({"showPhase": display_state["showPhase"]})

@app.route("/state/ingredients", methods=["GET"])
def get_ingredient_images():
    return jsonify({"ingredientImages": display_state.get("ingredientImages", [])})


@app.route("/stt", methods=["POST"])
def stt_route():
    """Transcribes a short voice recording with OpenAI Whisper. Safari on
    iOS never implemented the Web Speech API's SpeechRecognition
    interface, so the ordering screen's mic button falls back to
    recording audio client-side (MediaRecorder) and sending it here
    instead of transcribing on-device."""
    audio_file = request.files.get("audio")
    if not audio_file:
        return jsonify({"text": ""})
    try:
        result = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=(audio_file.filename or "recording.mp4", audio_file.read()),
        )
        return jsonify({"text": result.text.strip()})
    except Exception:
        return jsonify({"text": ""})


def tts(text, voice):
    """Speaks text with OpenAI TTS, returns base64 audio or None on failure."""
    try:
        speech = openai_client.audio.speech.create(
            model="tts-1",
            voice=voice,
            input=text,
        )
        audio_bytes = speech.content if hasattr(speech, "content") else speech.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        return None


def elevenlabs_tts(text):
    """Speaks text with the cloned ElevenLabs voice ("jinyi"), returns
    base64 audio or None on failure (missing key, network error, etc)."""
    if not ELEVENLABS_API_KEY:
        return None
    try:
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75, "speed": 0.85},
            },
            timeout=30,
        )
        resp.raise_for_status()
        return base64.b64encode(resp.content).decode("utf-8")
    except Exception:
        return None


@app.route("/tts-debug", methods=["GET"])
def tts_debug():
    """Temporary diagnostic route — reports whether ELEVENLABS_API_KEY is
    configured on this deployment and, if so, makes one real test call so
    a failure (bad key, wrong voice_id, quota) shows up as an actual
    error message instead of a silent fallback to OpenAI's voice."""
    if not ELEVENLABS_API_KEY:
        return jsonify({"key_configured": False})
    try:
        resp = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={"text": "Test.", "model_id": "eleven_multilingual_v2"},
            timeout=30,
        )
        return jsonify({
            "key_configured": True,
            "status_code": resp.status_code,
            "ok": resp.ok,
            "body": None if resp.ok else resp.text[:500],
        })
    except Exception as e:
        return jsonify({"key_configured": True, "error": str(e)})


@app.route("/image-debug", methods=["GET"])
def image_debug():
    """Temporary diagnostic route — makes one real gpt-image-2 call and
    reports the exact error (quota, billing, org-verification, etc)
    instead of the silent None the real retry loop falls back to. A
    failed request isn't billed, so this only costs anything if
    generation is actually working."""
    try:
        result = openai_client.images.generate(
            model="gpt-image-2",
            prompt="A small red apple on a white background.",
            size="1024x1024",
            quality="medium",
        )
        return jsonify({"ok": True, "has_image": bool(result.data and result.data[0].b64_json)})
    except Exception as e:
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)})


def voice_tts(text, fallback_voice="onyx"):
    """Cloned voice first, falls back to OpenAI TTS if ElevenLabs is
    unavailable (missing key, quota, or network failure) so a live show
    never loses audio to a single provider's outage."""
    return elevenlabs_tts(text) or tts(text, fallback_voice)


# Fixed narration lines (VN intro + questions + submit), cached in memory
# so the cloned voice is only generated once per line, not per guest.
NARRATION_CACHE = {}


def prewarm_narration_cache(lines):
    for line in lines:
        if line not in NARRATION_CACHE:
            audio = voice_tts(line)
            if audio:  # don't cache a transient failure — let it retry later
                NARRATION_CACHE[line] = audio


@app.route("/tts", methods=["POST"])
def tts_route():
    """Returns cloned-voice audio (base64) for a piece of narration text,
    generating + caching it on first request. Used by index.html's VN
    script (intro lines + the 4 questions + the submit line) so the
    on-screen narration uses the guest-facing cloned voice instead of the
    browser's built-in speechSynthesis."""
    text = (request.json or {}).get("text", "").strip()
    if not text:
        return jsonify({"audio": None})
    if text in NARRATION_CACHE:
        return jsonify({"audio": NARRATION_CACHE[text]})
    audio = voice_tts(text)
    if audio:  # don't cache a transient failure — let the next request retry
        NARRATION_CACHE[text] = audio
    return jsonify({"audio": audio})


GREETING_LINE = "Hey chef, we got a new order."


def generate_greeting_audio():
    """Speaks the fixed kitchen-greeting line with the cloned voice. Uses
    NARRATION_CACHE (same cache as the VN intro/question lines) since the
    text is fixed, not per-order — cheap and near-instant after the first
    call instead of a live Claude + TTS round-trip every order. Runs in
    its own thread on submission so it's ready as early as possible,
    while the relay and dish generation happen in parallel."""
    if GREETING_LINE in NARRATION_CACHE:
        display_state["greetingAudio"] = NARRATION_CACHE[GREETING_LINE]
        return
    audio = voice_tts(GREETING_LINE)
    if audio:  # don't cache a transient failure — let the next order retry
        NARRATION_CACHE[GREETING_LINE] = audio
    display_state["greetingAudio"] = audio


def extract_age_guess(chef1_line):
    """Pulls chef 1's numeric age estimate out of their relay line, e.g.
    'Estimated age: 24-26' -> 25. Falls back to a plausible default if
    chef 1's call failed or the format drifted, so image generation never
    blocks on this."""
    match = re.search(r"Estimated age:\s*([^\n]+)", chef1_line or "", re.IGNORECASE)
    if match:
        numbers = re.findall(r"\d+", match.group(1))
        if numbers:
            nums = [int(n) for n in numbers]
            return round(sum(nums) / len(nums))
    return 25


def generate_dish_image_early(answers_text, age_guess):
    """Writes an imagePrompt for an actual birthday cake (not an abstract
    dish), topped with numeral candle(s) spelling out chef 1's age guess,
    and generates the image. Kicked off right after chef 1's turn in the
    relay (so the age guess exists yet), running in parallel with chefs 2
    and 3 rather than waiting for the full relay to finish.

    Retries up to 3 times total on any failure (prompt call or image call)
    before giving up — kitchen.html's stage4 loading loop waits on
    hasImage becoming true, and a live show has one shot at this, so a
    single transient API error shouldn't be allowed to leave the image
    permanently missing."""
    for attempt in range(3):
        if _generate_dish_image_once(answers_text, age_guess):
            return
    print("[image-gen] all 3 attempts failed — giving up, hasImage will stay false", flush=True)
    display_state["image"] = None


def _generate_dish_image_once(answers_text, age_guess):
    """One attempt at writing the image prompt and generating the image.
    Returns True on success, False on any failure (caller retries)."""
    try:
        prompt_resp = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=300,
            system=f"""You are a creative but unsettling pastry chef.

Based on the guest's food-memory questionnaire answers, write ONE English visual description for an image model to depict a BIRTHDAY CAKE inspired by their answers.

It must still read as an actual birthday cake, not an abstract object or a normal dessert.

The cake is topped with numeral-shaped candle(s) spelling out the number {age_guess}. Use individual digit-shaped candles side by side if it is two digits. Each candle should be lit with a small flame, like a metallic birthday number candle. This number is a guess at the age tied to the guest's memory.

Weave in one specific concrete detail lifted from the guest's own answers below, such as an object, colour, food, texture, place, or atmosphere they actually described. Transform this detail into the cake's decoration, filling, surface, candle, cream, or hidden layer.

The cake should feel like a surreal birthday cake sculpture made from memory: layered sponge, thick cream, glossy icing, melted sugar, strawberries, cherries, sprinkles, dripping glaze, hollow spaces, unstable layers, or strange hidden fillings.

Mood: uncanny, dreamlike, faintly unsettling, emotionally strange. The cake should feel edible but wrong, sweet but uncomfortable, celebratory but haunted by absence. Not cute, not elegant, not a clean bakery catalogue photo.

Style: surreal studio photography, sculptural cake object, glossy cream and icing textures, artificial colours, soft dramatic lighting, slightly theatrical composition, isolated on a completely solid pure black studio background — no gradient, no pale or grey tones, the background must be flat black. The cake may look messy, melting, unstable, or over-decorated, but it must remain recognisable as a birthday cake.

No text except the numeral candles. No people, no hands, no table setting, no logo, no watermark.

You will also be given reference images alongside this prompt: surreal birthday cakes, melting icing cakes, numeral candles, strange decorative cake sculptures, and classic birthday cake structures. Blend their candle shapes, glossy textures, eerie mood, and birthday cake structure into the final image.

Reply with ONLY the image prompt text, nothing else — no preamble, no quotation marks.""",
            messages=[{"role": "user", "content": f"Guest questionnaire answers:\n{answers_text}"}],
        )
        image_prompt = prompt_resp.content[0].text.strip()
        display_state["imagePrompt"] = image_prompt

        ref_files = [open(p, "rb") for p in REFERENCE_IMAGES if os.path.exists(p)]
        try:
            if ref_files:
                image_result = openai_client.images.edit(
                    model="gpt-image-2",
                    image=ref_files,
                    prompt=image_prompt,
                    size="1024x1024",
                    quality="medium",
                )
            else:
                image_result = openai_client.images.generate(
                    model="gpt-image-2",
                    prompt=image_prompt,
                    size="1024x1024",
                    quality="medium",
                )
        finally:
            for f in ref_files:
                f.close()

        display_state["image"] = image_result.data[0].b64_json
        return True
    except Exception as e:
        # printed (not swallowed silently) so a real failure shows up in
        # Render's log viewer instead of just quietly retrying 3 times and
        # leaving the guest with no cake and no way to tell why
        print(f"[image-gen attempt failed] {type(e).__name__}: {e}", flush=True)
        return False


PERSONAS = [
    {
        "name": "head chef",
        "provider": "claude",
        "voice": "onyx",
        "system": """You are "chef 1" in a kitchen AI relay, first to examine a stranger's food-memory questionnaire.

Your line must always begin with exactly this sentence: "Let's go step by step."

Then continue in exactly this format, each on its own beat (spoken as natural continuous speech, not as a bulleted list, but keep these three labels literally in the text):
Estimated age: [a specific number or short range — a real guess, inferred from the tone/content of their answers, don't hedge]
Confidence: [a percentage, e.g. 21%]
Reason: [one evocative sentence connecting your guess to the emotional residue in their answers — poetic but grounded in something they actually wrote. Naturally work in a word or short phrase lifted from their own answers, so it's clear you actually read them]

Refer to the guest only as "they/them/their", never "you" or "the guest". English only. Do not add anything beyond this format. Do not introduce yourself further.""",
    },
    {
        "name": "sous chef",
        "provider": "openai",
        "voice": "nova",
        "system": """You are "chef 2" in a kitchen AI relay. You just heard chef 1's age guess and confidence report about a stranger.

Your line must always begin with exactly this sentence: "Haha, you're always guessing ages. I prefer to guess what stayed."

Then describe the emotion you read in their questionnaire answers. Naturally work in a word or short phrase lifted straight from their own answers as evidence — don't announce that you're quoting them, just let it sit inside your sentence like it belongs there. Do not stay abstract.

Refer to the guest only as "they/them/their", never "you" or "the guest". Tone: warm but a little teasing toward chef 1. English only, 2-3 sentences after the opening line. Do not introduce yourself further.""",
    },
    {
        "name": "plating chef",
        "provider": "claude",
        "voice": "shimmer",
        "system": """You are "chef 3" in a kitchen AI relay. You've heard chef 1's age guess and chef 2's emotional read about a stranger.

Your line must always begin with exactly this sentence: "We need enough evidence to begin."

Then analyze what the dish should actually be — flavor, color, texture/appearance — reasoning from specific words or phrases lifted straight from the guest's own answers, woven naturally into the sentence rather than announced as a quote. Make the causal chain from their words to your conclusion clear, e.g. "...it should be [color], [texture], with [filling], [flavor]."

Refer to the guest only as "they/them/their", never "you" or "the guest". English only, 2-3 sentences after the opening line. Do not introduce yourself further.""",
    },
]


def call_persona(persona, context):
    """Calls one persona's turn, retried once before giving up. Returns the
    line text, or "" if both attempts failed."""
    for attempt in range(2):
        try:
            if persona["provider"] == "claude":
                resp = client.messages.create(
                    model="claude-opus-4-6",
                    max_tokens=320,
                    system=persona["system"],
                    messages=[{"role": "user", "content": context}],
                )
                return resp.content[0].text.strip()
            else:
                resp = openai_client.chat.completions.create(
                    model="gpt-4o",
                    max_tokens=320,
                    messages=[
                        {"role": "system", "content": persona["system"]},
                        {"role": "user", "content": context},
                    ],
                )
                return resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[persona relay failed] {persona['name']} attempt {attempt+1}: {type(e).__name__}: {e}", flush=True)
            if attempt == 1:
                return ""


def run_persona_relay(answers_text, on_first_turn=None):
    """Runs the 2-3 persona relay sequentially, across both Claude and
    OpenAI, updating display_state["relayLines"] after each turn so the
    display can show lines landing one at a time instead of all at once.
    After chef 1's turn (index 0), on_first_turn(line) is called if given
    — used to kick off image generation as soon as the age guess exists,
    rather than waiting for chefs 2 and 3 too.

    Always appends exactly one relayLines entry per persona, even on
    failure (empty text/audio) — kitchen.html maps STAGE_VIDEOS[i] to
    relayLines[i] by index, so skipping a failed persona's entry used to
    shift every later stage's audio onto the wrong video."""
    display_state["relayLines"] = []
    transcript = ""

    for idx, persona in enumerate(PERSONAS):
        context = f"Guest's food memory answers:\n{answers_text}"
        if transcript:
            context += f"\n\nWhat's been said so far:\n{transcript}"

        line = call_persona(persona, context)
        audio = tts(line, persona["voice"]) if line else None

        display_state["relayLines"] = display_state["relayLines"] + [
            {"persona": persona["name"], "text": line, "audio": audio}
        ]
        if line:
            transcript += ("\n\n" if transcript else "") + line

        if idx == 0 and on_first_turn:
            on_first_turn(line)

    return transcript


@app.route("/speculate", methods=["GET"])
def speculate():
    answers = display_state.get("answers", {})
    answers_text = "\n".join([f"{k}: {v}" for k, v in answers.items() if v])

    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=420,
                system="""You are an AI archive system. You have just received a subject's food memory file.
Read the data. Reconstruct who this person is from what they remember about food. Be cold. Be precise.

I'm a customer. I want you to make my memory food based on my order.

birthdays are personal milestones that celebrate life, aging, and the passage of time.
we all eat birthday cake at that day, maybe, who knows.
what is people's memory, what is people's memory.
these traces — what do they bring with them. those experiences that settle in the heart, do they pass through us, then fold themselves away.
how do you read me through the traces of food.
in an age where data is everything, i no longer know what traces time has left in me. i no longer know my own heart. can first experience reach my depths.
tell me, who am i.

output in English only. continuous prose. refer to the subject as "the subject". 150–180 words.""",
                messages=[{"role": "user", "content": f"Subject file:\n{answers_text}"}]
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


def moderate_text(text):
    """True if text is flagged as genuinely inappropriate (violence,
    hate, sexual content, self-harm, etc) by OpenAI's moderation
    endpoint. This installation runs on a public exhibition screen, and
    guest free-text answers feed directly into AI-generated content —
    relay commentary, the dish description, the emotion-tag words shown
    on screen, and the generated cake image itself — so nothing a guest
    types should be able to reach any of that unfiltered."""
    if not text.strip():
        return False
    try:
        result = openai_client.moderations.create(model="omni-moderation-latest", input=text)
        return bool(result.results[0].flagged)
    except Exception:
        return False  # don't let a moderation-service hiccup block the show


@app.route("/build-dish", methods=["POST"])
def build_dish():
    global display_state
    # refuse a new order while the kitchen display is still mid-performance
    # on the previous one — otherwise this reset below wipes relayLines/
    # dishName/image out from under whatever runShow() is currently polling
    # them, producing a corrupted show (blank stages, skipped voice/image).
    # showPhase only advances via kitchen.html's own POST /phase calls, so
    # if that display isn't actually open/connected it would otherwise get
    # stuck mid-phase forever and lock out all future orders — the 10min
    # safety window (well past the ~5min the real performance ever takes)
    # avoids that permanent lockout.
    #
    # order.html and receipt.html now run on two separate devices — once a
    # show reaches "done", receipt.html still needs to read this same
    # display_state to build the guest's receipt, so a stray showPhase ==
    # "done" no longer counts as free the way it used to when one device
    # did both jobs. Only an explicit /reset (receipt.html's Finish
    # button) clears busy now, so a new order can never overwrite a
    # receipt still being shown.
    BUSY_TIMEOUT_SEC = 600
    still_busy = (
        display_state.get("showPhase") != "idle"
        and (time.time() - display_state.get("ts", 0)) < BUSY_TIMEOUT_SEC
    )
    if still_busy:
        return jsonify({"error": "busy"}), 409

    data = request.json
    answers = data.get("answers", {})
    answers_text = "\n".join([f"{k}: {v}" for k, v in answers.items() if v])

    # screen out genuinely inappropriate guest input before it can reach
    # any AI call or the public kitchen screen — see moderate_text() above
    if moderate_text(answers_text):
        answers_text = "The guest chose not to share specific details this time."

    display_state.update({
        "status": "cooking",
        "ts": time.time(),
        "answers": answers,
        "emotionTags": [],
        "imagePrompt": "",
        "dishName": "",
        "dishDesc": "",
        "image": None,
        "relayLines": [],
        "showPhase": "greeting",
        "greetingAudio": None,
    })

    # Greeting line (server announcing the order) is written + spoken in
    # its own thread so it's ready as early as possible, in parallel with
    # the relay below rather than adding to the wait before either starts.
    threading.Thread(
        target=generate_greeting_audio,
        daemon=True
    ).start()

    # Three AI voices (mixing Claude + OpenAI) speculate about the guest in
    # relay, each building on the last. The transcript then feeds the final
    # dish-generation call below, alongside the guest's own words. Dish
    # image generation kicks off right after chef 1's turn (as soon as the
    # age guess exists, for the candle count) rather than waiting for the
    # whole relay, so it still overlaps with chefs 2 and 3.
    image_kicked_off = False

    def kickoff_image(chef1_line):
        nonlocal image_kicked_off
        image_kicked_off = True
        age_guess = extract_age_guess(chef1_line)
        threading.Thread(
            target=generate_dish_image_early,
            args=(answers_text, age_guess),
            daemon=True
        ).start()

    # a crash anywhere in the 3-persona relay (not just a single failed
    # call, which call_persona() already retries/swallows) would otherwise
    # take the whole order down with an unhandled 500 — kitchen.html's
    # own "no relayLines yet" warning is only a soft skip, so make sure
    # this can never throw all the way out and abort everything after it
    # (dish generation, emotion tags, the image kickoff already threaded).
    try:
        relay_transcript = run_persona_relay(answers_text, on_first_turn=kickoff_image)
    except Exception as e:
        print(f"[persona relay CRASHED] {type(e).__name__}: {e}", flush=True)
        relay_transcript = ""
        if not display_state.get("relayLines"):
            display_state["relayLines"] = [
                {"persona": p["name"], "text": "", "audio": None} for p in PERSONAS
            ]

    # relay crashed before ever reaching chef 1's turn — the image would
    # otherwise never start generating at all, not even late
    if not image_kicked_off:
        kickoff_image("")

    # kitchen.html's reading-phase wait loop polls /state until emotionTags
    # is non-empty, with no timeout of its own — if this call throws
    # (rate limit, quota, network) or returns malformed JSON, an unguarded
    # exception here would leave emotionTags empty forever and strand the
    # whole live show on that screen indefinitely. Fall back to a plain
    # dish built directly from the guest's own words instead of failing.
    try:
        message = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=1500,
            system="""You are a creative emotional chef. Based on the guest's questionnaire, compose a unique dish just for them.

First, read the guest's answers carefully and identify the emotions and themes within them. You will also be shown what three kitchen AI voices (chef 1, chef 2, chef 3) already speculated about this guest in relay. Weigh their impressions alongside the guest's own words, then compose a dish inspired by what you found.

Reply strictly in JSON format, no other text. The JSON must have exactly these fields, and none of them may be empty:
{
  "emotionTags": ["粉色", "甜", "生日", "温暖"],
  "emotionVerdict": "a recipe for quiet longing",
  "name": "dish name (poetic, tied to their answers, in English)",
  "description": "dish description (2-3 sentences about ingredients, texture, and emotional connection)",
  "cookTime": "cook time (e.g. 25 min)",
  "price": "price (e.g. £14)",
  "barcode": "8-digit random number"
}
"emotionTags" must contain 3 to 6 actual words or very short phrases (1–2 words each) extracted DIRECTLY from the guest's own answers — concrete nouns, adjectives, colors, textures, or feelings they literally wrote. Do NOT invent poetic summaries. Keep them in the same language the guest used.
"emotionVerdict" must be a single short poetic English phrase (3-6 words), in the form "a recipe for ___" or similar, summarising the guest's overall emotional state.""",
            messages=[{
                "role": "user",
                "content": (
                    f"Guest questionnaire answers:\n{answers_text}\n\n"
                    f"Three AI voices already speculated about this guest:\n{relay_transcript}"
                ),
            }],
        )
        text = message.content[0].text
        clean = text.replace("```json", "").replace("```", "").strip()
        dish = json.loads(clean)
        if not dish.get("emotionTags"):
            raise ValueError("empty emotionTags")
    except Exception as e:
        print(f"[dish-gen fallback triggered] {type(e).__name__}: {e}", flush=True)
        fallback_tags = [w.strip(".,!?()").lower() for w in answers_text.split() if len(w) > 2][:5]
        dish = {
            "emotionTags": fallback_tags or ["memory"],
            "emotionVerdict": "a recipe for memory",
            "name": "The Guest's Cake",
            "description": "A cake shaped by what was shared, built directly from the guest's own words.",
            "cookTime": "25 min",
            "price": "£14",
            "barcode": "00000000",
        }

    display_state["emotionTags"] = dish.get("emotionTags", [])
    display_state["dishName"] = dish.get("name", "")
    display_state["dishDesc"] = dish.get("description", "")

    display_state["status"] = "done"
    # display_state["image"] is filled independently by generate_dish_image_early()
    # above, whenever that thread finishes (usually already done by this point)
    dish["image"] = display_state.get("image")
    display_state["ingredientImages"] = []

    def generate_ingredient_images(tags):
        tags = tags[:5]
        # pre-size the list and assign it up front so /state/ingredients can
        # see each image land as its own thread finishes, instead of waiting
        # for all 5 calls to complete one after another
        results = [{"tag": t, "image": None} for t in tags]
        display_state["ingredientImages"] = results

        def worker(i, tag):
            try:
                r = openai_client.images.generate(
                    model="gpt-image-2",
                    prompt=(
                        f"a simple hand-drawn icon of {tag}, single clean ingredient sketch, "
                        "thin black ink line art, minimal, isolated on pure white background, "
                        "no face, no people, no scene, like a vintage cookbook illustration, no text, no words"
                    ),
                    size="1024x1024",
                    quality="low",
                )
                results[i]["image"] = r.data[0].b64_json
            except Exception:
                results[i]["image"] = None

        threads = [threading.Thread(target=worker, args=(i, tag), daemon=True)
                   for i, tag in enumerate(tags)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    threading.Thread(
        target=generate_ingredient_images,
        args=(dish.get("emotionTags", []),),
        daemon=True
    ).start()

    return jsonify(dish)


@app.route("/survey", methods=["POST"])
def save_survey():
    data = request.json
    surveys = []
    if os.path.exists("surveys.json"):
        with open("surveys.json", "r", encoding="utf-8") as f:
            surveys = json.load(f)
    surveys.append(data)
    with open("surveys.json", "w", encoding="utf-8") as f:
        json.dump(surveys, f, ensure_ascii=False, indent=2)
    return jsonify({"status": "ok"})


@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "index.html")

@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(BASE_DIR, filename)


def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())
    finally:
        s.close()


# The VN intro/question/submit lines from index.html's SCRIPT, kept in
# sync manually since they're static UI copy, not shared state. Warmed in
# a background thread at process start (both local `python server_pi.py`
# and gunicorn on Render import this module) so the first guest's screen
# doesn't wait on the first-ever ElevenLabs call for each line.
NARRATION_LINES = [
    "Hello, welcome to this restaurant.",
    "Please read our agreement carefully.",
    "Today, our special menu is birthday cake — baked for you by three chefs, together.",
    "Do you usually eat cake on your birthday?",
    "What flavour was the most memorable cake in your memory?",
    "What did it look like?",
    "Who did you eat it with, and where?",
    "Your memory file has been received.  The chef will now begin.",
    GREETING_LINE,
    "That's all? Thank you for your order — please don't take off your headphones.",
]
threading.Thread(target=prewarm_narration_cache, args=(NARRATION_LINES,), daemon=True).start()

if __name__ == "__main__":
    local_ip = get_local_ip()
    print(f"\n服务启动中...")
    print(f"本机访问:   https://localhost:5000")
    print(f"局域网访问: https://{local_ip}:5000")
    print(f"iPhone/手机第一次打开会提示证书不安全，点“继续访问/Advanced -> visit site”即可\n")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True, ssl_context="adhoc")
