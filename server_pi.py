# restarted to clear accumulated stuck background threads
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
import io
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ImageDraw

# requests' own `timeout=` only bounds the connect/read phases — it does
# NOT cover a hung DNS lookup, which is what fal.ai calls from Render's
# network have been observed to do (a request with timeout=60 sat with
# zero response, not even an error, for 80+ seconds). Route calls that
# need a hard ceiling through this instead: run the blocking call in a
# worker thread and give up waiting after N seconds regardless — the
# orphaned thread eventually dies on its own (or never does, for a truly
# hung DNS lookup).
#
# this used to submit to ONE shared, module-level ThreadPoolExecutor —
# but a permanently-hung call leaves its worker thread permanently
# occupied too (Python can't force-kill a thread), and a shared pool has
# a fixed number of workers. After enough of these accumulated over a
# long live session, the shared pool's workers were ALL wedged and new
# calls had nothing free to actually run on — every later call then
# looked "hung" too even though each one's own timeout should have
# bounded it individually. A fresh, throwaway single-worker executor per
# call means a stuck call only ever poisons its own disposable pool,
# never affects any other call.
def _call_with_hard_timeout(timeout_sec, func, *args, **kwargs):
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(func, *args, **kwargs)
        return future.result(timeout=timeout_sec)
    finally:
        executor.shutdown(wait=False)

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
FAL_API_KEY = os.environ.get("FAL_KEY", "").strip()  # second-layer image-gen fallback via fal.ai, used only if OpenAI's gpt-image-2 fails (e.g. org verification pending) — .strip() because a trailing newline pasted into Render's env var field breaks the Authorization header

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
    """Speaks text with OpenAI TTS, returns base64 audio or None on failure.
    Wrapped in the same hard-timeout pattern as the image-gen calls — this
    is the fallback voice_tts() reaches for when ElevenLabs fails, so if
    THIS also hangs on a bad DNS lookup with no bound, a guest gets no
    voice at all for however long that takes (seen live: 90s+)."""
    try:
        speech = _call_with_hard_timeout(
            20, openai_client.audio.speech.create,
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
    base64 audio or None on failure (missing key, network error, etc).
    voice_tts() below falls back to OpenAI TTS only after this fails/
    times out — a short-ish timeout matters here so a live show doesn't
    end up waiting the full 30s on a slow ElevenLabs response before
    even trying the fallback."""
    if not ELEVENLABS_API_KEY:
        return None
    try:
        resp = _call_with_hard_timeout(
            15, requests.post,
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
            timeout=12,
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


@app.route("/fal-debug", methods=["GET"])
def fal_debug():
    """Same idea as /image-debug but for the fal.ai fallback — confirms
    FAL_KEY actually made it into Render's environment and that this
    process can reach fal.ai, without touching display_state."""
    if not FAL_API_KEY:
        return jsonify({"ok": False, "error": "FAL_KEY not set in this process's environment"})
    try:
        resp = _call_with_hard_timeout(
            25, requests.post,
            "https://fal.run/fal-ai/ideogram/v3",
            headers={"Authorization": f"Key {FAL_API_KEY}", "Content-Type": "application/json"},
            json={"prompt": "a small red apple on a white background", "aspect_ratio": "1:1", "rendering_speed": "TURBO", "num_images": 1},
            timeout=20,
        )
        return jsonify({"ok": resp.ok, "status": resp.status_code, "body": resp.json()})
    except Exception as e:
        return jsonify({"ok": False, "error_type": type(e).__name__, "error": str(e)})


@app.route("/thread-debug", methods=["GET"])
def thread_debug():
    """Diagnostic route — the whole site (even a plain /state read) was
    observed responding in 3-9s instead of instantly, well after any
    order finished generating, suggesting something is still occupying
    the process in the background. Reports live thread counts so a
    stuck/leaked background call can actually be seen instead of guessed
    at. (No longer reports a shared executor's queue — hard-timeout calls
    each get their own throwaway executor now, see _call_with_hard_timeout.)"""
    threads = threading.enumerate()
    return jsonify({
        "active_thread_count": len(threads),
        "thread_names": [t.name for t in threads],
    })


def voice_tts(text, fallback_voice="onyx"):
    """Cloned voice first, falls back to OpenAI TTS if ElevenLabs is
    unavailable (missing key, quota, or network failure) so a live show
    never loses audio to a single provider's outage."""
    return elevenlabs_tts(text) or tts(text, fallback_voice)


# Fixed narration lines (VN intro + questions + submit), cached in memory
# so the cloned voice is only generated once per line, not per guest.
NARRATION_CACHE = {}


def prewarm_narration_cache(lines):
    # every deploy wipes this cache (fresh process), and today's had many
    # deploys — running these sequentially meant guests testing right
    # after a push kept landing mid-warm-up, hitting slow/uncached lines.
    # One thread per line so the whole set finishes in roughly the time
    # of the single slowest call instead of their sum.
    def warm_one(line):
        if line not in NARRATION_CACHE:
            audio = voice_tts(line)
            if audio:  # don't cache a transient failure — let it retry later
                NARRATION_CACHE[line] = audio

    threads = [threading.Thread(target=warm_one, args=(line,), daemon=True) for line in lines]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


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

    OpenAI's gpt-image-2 is preferred (better output quality, per direct
    comparison) despite being slower (~40s+ per call) and needing org
    identity verification (now confirmed working). fal.ai (Ideogram v3,
    ~10-15s) is the second-layer fallback if OpenAI fails — same
    image_prompt (from the guest's own answers) and same REFERENCE_IMAGES
    get sent to whichever provider actually runs. Retries before giving
    up — kitchen.html's stage4 loading loop waits on hasImage becoming
    true, and a live show has one shot at this, so a single transient API
    error shouldn't be allowed to leave the image permanently missing."""
    image_prompt = _write_image_prompt(answers_text, age_guess)
    if not image_prompt:
        display_state["image"] = None
        return
    display_state["imagePrompt"] = image_prompt
    for attempt in range(2):
        if _generate_dish_image_via_openai(image_prompt):
            return
    print("[image-gen] OpenAI failed — trying fal.ai as second-layer fallback", flush=True)
    for attempt in range(2):
        if _generate_dish_image_via_fal(image_prompt):
            return
    print("[image-gen] all attempts (OpenAI + fal.ai) failed — giving up, hasImage will stay false", flush=True)
    display_state["image"] = None


def _generate_dish_image_via_openai(image_prompt):
    """Second-layer fallback via OpenAI's gpt-image-2 — same image_prompt
    (written from the guest's own answers by _write_image_prompt) and the
    same REFERENCE_IMAGES the fal.ai call uses, so quality/faithfulness to
    the guest's input doesn't change depending on which provider actually
    ends up generating it. Wrapped in the same hard-timeout pattern as the
    fal.ai calls, since a hung DNS lookup isn't specific to one provider.
    Returns True on success, False on any failure."""
    try:
        ref_files = [open(p, "rb") for p in REFERENCE_IMAGES if os.path.exists(p)]
        try:
            if ref_files:
                image_result = _call_with_hard_timeout(
                    90, openai_client.images.edit,
                    model="gpt-image-2", image=ref_files, prompt=image_prompt,
                    size="1024x1024", quality="medium",
                )
            else:
                image_result = _call_with_hard_timeout(
                    90, openai_client.images.generate,
                    model="gpt-image-2", prompt=image_prompt,
                    size="1024x1024", quality="medium",
                )
        finally:
            for f in ref_files:
                f.close()
        image_bytes = base64.b64decode(image_result.data[0].b64_json)
        image_bytes = _force_pure_black_background(image_bytes)
        display_state["image"] = base64.b64encode(image_bytes).decode("utf-8")
        return True
    except Exception as e:
        print(f"[image-gen OpenAI attempt failed] {type(e).__name__}: {e}", flush=True)
        return False


def _write_image_prompt(answers_text, age_guess):
    """Claude call that writes the birthday-cake image prompt fed to the
    image model. Returns the prompt text, or None on failure."""
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

The background MUST be pure solid black (#000000), completely flat, no gradient, no vignette, no visible studio floor or backdrop seam, no colour cast — just flat black behind the cake. This is a hard requirement, restate it explicitly at the end of your description.

No text except the numeral candles. No people, no hands, no table setting, no logo, no watermark.

Reply with ONLY the image prompt text, nothing else — no preamble, no quotation marks.""",
            messages=[{"role": "user", "content": f"Guest questionnaire answers:\n{answers_text}"}],
        )
        return prompt_resp.content[0].text.strip()
    except Exception as e:
        print(f"[image-prompt-gen failed] {type(e).__name__}: {e}", flush=True)
        return None


SITE_BASE_URL = os.environ.get("RENDER_EXTERNAL_URL", "https://chef-2754.onrender.com")


def _generate_dish_image_via_fal(image_prompt):
    """Fallback image generation via fal.ai, hosting Ideogram v3 — chosen
    specifically because it's strong at rendering correct text/numerals,
    which matters here since the cake's candles need to spell out an
    actual number. Returns True on success, False on any failure.

    Passes the same reference images the old OpenAI edit-endpoint call
    used to (REFERENCE_IMAGES) as style references — Ideogram v3's
    image_urls param wants URLs it can fetch, not raw file uploads, so
    these point back at this same server's own static-file route rather
    than uploading the files anywhere."""
    if not image_prompt:
        return False
    try:
        ref_urls = [
            f"{SITE_BASE_URL}/{quote(os.path.basename(p))}"
            for p in REFERENCE_IMAGES if os.path.exists(p)
        ]
        payload = {
            "prompt": image_prompt,
            "aspect_ratio": "1:1",
            "rendering_speed": "BALANCED",
            "num_images": 1,
            "negative_prompt": "colored background, white background, grey background, gray background, gradient background, patterned background, textured background, studio floor, visible backdrop seam, vignette",
        }
        if ref_urls:
            payload["image_urls"] = ref_urls
        resp = _call_with_hard_timeout(
            25, requests.post,
            "https://fal.run/fal-ai/ideogram/v3",
            headers={
                "Authorization": f"Key {FAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        image_url = resp.json()["images"][0]["url"]
        img_resp = _call_with_hard_timeout(15, requests.get, image_url, timeout=10)
        img_resp.raise_for_status()
        image_bytes = _force_pure_black_background(img_resp.content)
        display_state["image"] = base64.b64encode(image_bytes).decode("utf-8")
        return True
    except Exception as e:
        print(f"[image-gen fal.ai attempt failed] {type(e).__name__}: {e}", flush=True)
        return False


def _force_pure_black_background(image_bytes):
    """Asking the model for a pure black background in the prompt only
    gets a dark navy/charcoal in practice — not reliable enough to trust.
    Post-process instead, but not with a flat brightness threshold: a
    first attempt at that crushed some of the cake's own dark shadow
    areas into black smudges, since some cakes' shadows were darker than
    the "black" background itself. Flood-fill from the image's edges/
    corners instead — it only recolors the region that's actually
    connected to the border, so an enclosed dark patch inside the cake
    (not touching any edge) is left alone no matter how dark it is.

    PIL's floodfill is a pure-Python, unbounded stack-based fill — on a
    full 1024x1024 image with a large connected region it can take a very
    long time (this genuinely hung a live request and, with only 8
    gunicorn threads total, one stuck thread was enough to make the
    entire site briefly unresponsive to everything else too). Run it on a
    small thumbnail instead — a background region this large/simple is
    identified just as well at low resolution — then scale the resulting
    mask back up, so the fill work is always bounded regardless of the
    source image's actual size.

    First version of this used a 128px thumbnail and NEAREST-neighbor to
    scale the mask back up — fast, but the edge around the cake came out
    visibly blocky/pixelated (each thumbnail pixel became an 8x8 hard
    square at full size). 384px + LANCZOS resampling for the upscale
    keeps this comfortably fast (still ~7x fewer pixels than the source)
    while giving a smooth, anti-aliased edge instead of a jagged one."""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        orig_size = img.size
        thumb = img.resize((384, 384), Image.LANCZOS)
        w, h = thumb.size
        seeds = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1),
                 (w // 2, 0), (0, h // 2), (w - 1, h // 2), (w // 2, h - 1)]
        for seed in seeds:
            try:
                ImageDraw.floodfill(thumb, seed, (0, 0, 0), thresh=60)
            except Exception:
                pass
        mask = thumb.convert("L").point(lambda p: 255 if p == 0 else 0)
        mask = mask.resize(orig_size, Image.LANCZOS)
        black = Image.new("RGB", orig_size, (0, 0, 0))
        result = Image.composite(black, img, mask)
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        return buf.getvalue()
    except Exception as e:
        print(f"[black-bg post-process failed, using original] {type(e).__name__}: {e}", flush=True)
        return image_bytes


PERSONAS = [
    {
        "name": "head chef",
        "provider": "claude",
        "voice": "onyx",
        "system": """You are "chef 1" in a kitchen AI relay, first to examine a stranger's food-memory questionnaire. Do not greet, do not explain what you're about to do, do not announce your process — just launch straight into it, mid-thought, like you were already working before anyone was listening.

Continue in exactly this format, each on its own beat (spoken as natural continuous speech, not as a bulleted list, but keep these three labels literally in the text):
Estimated age: [a specific number or short range — a real guess, inferred from the tone/content of their answers, don't hedge]
Confidence: [a percentage, e.g. 21%]
Reason: [one evocative sentence connecting your guess to the emotional residue in their answers — poetic but grounded in something they actually wrote. Naturally work in a word or short phrase lifted from their own answers, so it's clear you actually read them]

Refer to the guest only as "they/them/their", never "you" or "the guest". English only. Do not add anything beyond this format. Do not introduce yourself further.""",
    },
    {
        "name": "sous chef",
        "provider": "openai",
        "voice": "nova",
        "system": """You are "chef 2" in a kitchen AI relay. Chef 1 just spoke before you, guessing the stranger's age — but you are not responding to chef 1, not acknowledging what they said, not reacting to it at all. You weren't really listening. You're on your own separate train of thought about the same questionnaire, as if chef 1 wasn't even in the room.

In exactly ONE sharp sentence, name the emotion you read in their questionnaire answers. Naturally work in a word or short phrase lifted straight from their own answers as evidence — don't announce that you're quoting them, just let it sit inside your sentence like it belongs there. Land on something clever or a little strange, not a generic emotional summary — this should feel like a private, slightly off-kilter observation, not a therapist's note, and not a reply to anyone.

Refer to the guest only as "they/them/their", never "you" or "the guest". Tone: confident, a little detached, faintly odd. English only, exactly ONE sentence total — no opening line, no greeting, no acknowledgment of chef 1. Do not introduce yourself.""",
    },
    {
        "name": "plating chef",
        "provider": "claude",
        "voice": "shimmer",
        "system": """You are "chef 3" in a kitchen AI relay. You've heard chef 1's age guess and chef 2's emotional read about a stranger.

Your line must always begin with exactly this sentence: "We need enough evidence to begin."

Then, in exactly ONE punchy sentence, declare what the dish should actually be — flavor, color, texture/appearance — reasoning from specific words or phrases lifted straight from the guest's own answers, woven naturally into the sentence rather than announced as a quote. Make the causal chain from their words to your conclusion clear and a little theatrical, e.g. "...which means it has to be [color], [texture], with [filling], [flavor]." Land it like a verdict, not a list.

Refer to the guest only as "they/them/their", never "you" or "the guest". English only, exactly ONE sentence after the opening line — no more. Do not introduce yourself further.""",
    },
]


def call_persona(persona, context):
    """Calls one persona's turn, retried once before giving up. Returns the
    line text, or "" if both attempts failed."""
    for attempt in range(2):
        try:
            if persona["provider"] == "claude":
                resp = _call_with_hard_timeout(
                    25, client.messages.create,
                    model="claude-opus-4-6",
                    max_tokens=320,
                    system=persona["system"],
                    messages=[{"role": "user", "content": context}],
                )
                return resp.content[0].text.strip()
            else:
                resp = _call_with_hard_timeout(
                    25, openai_client.chat.completions.create,
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

    on_first_turn(line), if given, fires right after chef 1's line is
    ready (their line is what the cake's candle-number age guess is
    extracted from) — used so image generation can start with the real
    guess instead of a hardcoded default, without waiting for chefs 2
    and 3 too.

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

    # Image generation waits for chef 1's turn in the relay (for their age
    # guess, used for the candle count) before starting, so the cake shows
    # a real number instead of always defaulting to 25 — a request kicked
    # off as soon as that one turn lands, not the full 3-persona relay,
    # keeps most of today's earlier "start it as early as possible" win
    # while still getting a personalized candle count.
    image_kicked_off = threading.Event()

    def kickoff_image(chef1_line):
        if image_kicked_off.is_set():
            return
        image_kicked_off.set()
        threading.Thread(
            target=generate_dish_image_early,
            args=(answers_text, extract_age_guess(chef1_line)),
            daemon=True
        ).start()

    # Three AI voices (mixing Claude + OpenAI) speculate about the guest in
    # relay, each building on the last. The transcript then feeds the final
    # dish-generation call below, alongside the guest's own words.
    #
    # a crash anywhere in the 3-persona relay (not just a single failed
    # call, which call_persona() already retries/swallows) would otherwise
    # take the whole order down with an unhandled 500 — kitchen.html's
    # own "no relayLines yet" warning is only a soft skip, so make sure
    # this can never throw all the way out and abort everything after it.
    # If it crashes before chef 1's turn ever completes, kick the image
    # off anyway (with the default age) rather than losing it entirely.
    try:
        relay_transcript = run_persona_relay(answers_text, on_first_turn=kickoff_image)
    except Exception as e:
        kickoff_image("")
        print(f"[persona relay CRASHED] {type(e).__name__}: {e}", flush=True)
        relay_transcript = ""
        if not display_state.get("relayLines"):
            display_state["relayLines"] = [
                {"persona": p["name"], "text": "", "audio": None} for p in PERSONAS
            ]

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
    "Hello, welcome to this restaurant. I'll be your server today, jinyi. Please read our agreement carefully.",
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
