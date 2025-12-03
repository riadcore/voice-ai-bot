import os
import json
import re
import uuid
from pydub import AudioSegment
from num2words import num2words
from datetime import datetime
from TTS.api import TTS 
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    Response,
    jsonify,
)
from dotenv import load_dotenv
import random 



from groq import Groq
from signalwire.rest import Client as SignalWireClient
from signalwire.voice_response import VoiceResponse, Gather

# -------------------------------------------------
# Environment & basic setup
# -------------------------------------------------
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev_secret")


# Groq / LLaMA
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLAMA_MODEL = "llama-3.1-8b-instant"

groq_client = Groq(api_key=GROQ_API_KEY)

# ---- Coqui Bangla TTS model (offline) ----
os.environ["COQUI_TOS_AGREED"] = "1"   # Required for Coqui models

ELEVEN_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVEN_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")


BN_MODEL_NAME = "tts_models/bn/custom/vits-female"

print("Loading Coqui Bangla TTS model (vits-female)... This may take a minute the first time.")

bn_tts = TTS(BN_MODEL_NAME)


# SignalWire
SIGNALWIRE_PROJECT_ID = os.getenv("SIGNALWIRE_PROJECT_ID")
SIGNALWIRE_API_TOKEN = os.getenv("SIGNALWIRE_API_TOKEN")
SIGNALWIRE_SPACE_URL = os.getenv("SIGNALWIRE_SPACE_URL")
SIGNALWIRE_CALLER_ID = os.getenv("SIGNALWIRE_CALLER_ID")

signalwire_client = SignalWireClient(
    SIGNALWIRE_PROJECT_ID,
    SIGNALWIRE_API_TOKEN,
    signalwire_space_url=SIGNALWIRE_SPACE_URL,
)

BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")

# In-memory storage (replace with DB later if you want)
orders = {}
NEXT_ORDER_ID = 1


# -------------------------------------------------
# Helpers: normalize phone, call LLaMA, scripts
# -------------------------------------------------
def normalize_phone_bd(raw: str) -> str | None:
    """
    Normalize Bangladesh numbers into E.164 for SignalWire.
    Examples:
      '01712-345678'    -> '+8801712345678'
      '+8801712345678'  -> '+8801712345678'
      '8801712345678'   -> '+8801712345678'
      '1712345678'      -> '+8801712345678'
    """
    if not raw:
        return None

    digits = re.sub(r"\D", "", raw)

    # Already has country code 880
    if digits.startswith("880") and len(digits) == 13:
        return "+" + digits           # '88017...' -> '+88017...'

    # Local 11-digit mobile starting with 0, e.g. 017xxxxxxxx
    if digits.startswith("0") and len(digits) == 11:
        return "+88" + digits         # '017...' -> '+88017...'

    # 10-digit without leading 0, e.g. 1712345678
    if digits.startswith("1") and len(digits) == 10:
        return "+880" + digits        # '171...' -> '+880171...'

    # If original had a '+' and enough digits, keep it
    if raw.strip().startswith("+") and len(digits) >= 11:
        return "+" + digits

    return None



def parse_order_with_llama(order_text: str) -> dict:
    """
    Ask LLaMA (via Groq) to parse a free-form Bangla/English order
    into structured JSON.
    """
    system_prompt = (
        "You are an assistant that extracts structured order data "
        "from free-text Bangla or English messages about shirt orders. "
        "Always respond with valid JSON ONLY, no explanation."
    )

    user_prompt = f"""
Customer message (Bangla / English mixed):

\"\"\"{order_text}\"\"\"


Important:
- The customer is from Bangladesh.
- Mobile numbers usually look like: 017xxxxxxxx, 018xxxxxxxx, 019xxxxxxxx, or with country code 880 / +880.
- Always try to extract a phone number if there are 10–14 digits that look like a Bangladeshi mobile.
- Return the phone number as a string in whatever format appears (e.g. "01712345678" or "+8801712345678").



Extract:
- customer_name (if present)
- quantity (number of shirts)
- color
- size (or sizes list)
- price_total (numeric, if mentioned)
- phone
- address
- any other_notes

Return JSON with keys:
customer_name, quantity, color, size, price_total, phone, address, other_notes.
If something not found, use null.
"""

    chat = groq_client.chat.completions.create(
        model=LLAMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        response_format={"type": "json_object"},
    )

    raw = chat.choices[0].message.content
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {
            "customer_name": None,
            "quantity": None,
            "color": None,
            "size": None,
            "price_total": None,
            "phone": None,
            "address": None,
            "other_notes": raw,
        }

    return data


def build_bangla_script(parsed: dict) -> str:
    """
    Build Bangla confirmation script spoken to the customer.
    প্রথমে বট নিজের পরিচয় দেবে, তারপর অর্ডার রিক্যাপ করবে।
    """
    # Parsed values from LLaMA
    name = parsed.get("customer_name") or "স্যার"
    qty = parsed.get("quantity") or "একটি"
    color = parsed.get("color") or "শার্ট"
    size = parsed.get("size") or ""
    price = parsed.get("price_total")
    addr = parsed.get("address") or "আপনার দেওয়া ঠিকানায়"

    # Quantity → text
    if isinstance(qty, (int, float)):
        qty_text = f"{int(qty)} টি"
    else:
        qty_text = str(qty)

    size_part = ""
    if size:
        size_part = f", সাইজ {size}"

    price_part = ""
    if price:
        price_part = f", মোট মূল্য {price} টাকা"

    # 🔹 Intro part – exactly what you asked for
    intro = (
        "আসসালামু আলাইকুম। আমি একজন অটোমেশন বট কথা বলছি। "
        "আপনি একটি শার্ট অর্ডার করেছেন। "
        "আমি এখন আপনার অর্ডারটির কনফার্মেশন নেব। "
    )

    # 🔹 Order recap part
    recap = (
        f"{name}, আপনি {qty_text} {color}{size_part} অর্ডার করেছেন{price_part}. "
        f"ডেলিভারি হবে {addr}. "
    )

    # 🔹 Clear instruction for the customer
    ask_details = (
        "এখন দয়া করে পরিষ্কার করে বলবেন – "
        "শার্টের মডেল, রঙ এবং সাইজ ঠিক আছে কিনা, "
        "আর অর্ডারটি কনফার্ম করতে চান নাকি ক্যান্সেল করতে চান। "
        "যদি অর্ডারটি ঠিক থাকে, বলবেন – ‘হ্যাঁ, অর্ডার কনফার্ম’। "
        "যদি বাতিল করতে চান, বলবেন – ‘না, অর্ডার ক্যান্সেল’। "
        "এখন আপনার সিদ্ধান্ত বলুন।"
    )

    script = intro + recap + ask_details
    return script



def classify_customer_reply(text: str) -> str:
    """
    Very simple Bangla classification without LLM:
    returns 'confirmed', 'cancelled', or 'unclear'.
    """
    if not text:
        return "unclear"

    t = text.lower()

    # Common Bangla/English confirm patterns
    if any(
        phrase in t
        for phrase in [
            "হ্যাঁ",
            "ঠিক আছে",
            "কনফার্ম",
            "confirm",
            "হ্যা",
        ]
    ):
        # avoid cases like "না, কনফার্ম না"
        if "না" in t and "কনফার্ম" in t:
            return "cancelled"
        return "confirmed"

    if any(
        phrase in t
        for phrase in [
            "না",
            "ক্যান্সেল",
            "cancel",
            "চাই না",
            "বাতিল",
        ]
    ):
        return "cancelled"

    return "unclear"


# -------------------------------------------------
# Web UI routes
# -------------------------------------------------



@app.route("/", methods=["GET", "POST"])
def index():
    global NEXT_ORDER_ID

    if request.method == "POST":
        raw_text = request.form.get("order_text", "").strip()
        phone_manual = request.form.get("phone_manual", "").strip()

        if not raw_text:
            flash("Please paste the customer message first.", "error")
            return redirect(url_for("index"))

        parsed = parse_order_with_llama(raw_text)

        # If user typed phone manually, override AI result
        if phone_manual:
            parsed["phone"] = phone_manual

        script = build_bangla_script(parsed)

        order_id = NEXT_ORDER_ID

        NEXT_ORDER_ID += 1

        orders[order_id] = {
            "id": order_id,
            "raw_text": raw_text,
            "parsed": parsed,
            "script": script,
            "status": "pending",
            "created_at": datetime.utcnow(),
            "last_call_sid": None,
            "last_result": None,
        }

        return redirect(url_for("order_detail", order_id=order_id))

    return render_template("index.html", orders=list(orders.values()))


@app.route("/order/<int:order_id>")
def order_detail(order_id):
    order = orders.get(order_id)
    if not order:
        flash("Order not found.", "error")
        return redirect(url_for("index"))
    return render_template("order_detail.html", order=order)


# -------------------------------------------------
# Start SignalWire outbound call
# -------------------------------------------------
@app.route("/start_call/<int:order_id>", methods=["POST"])
def start_call(order_id):
    order = orders.get(order_id)
    if not order:
        flash("Order not found.", "error")
        return redirect(url_for("index"))

    raw_phone = order["parsed"].get("phone")
    phone = normalize_phone_bd(raw_phone)

    if not phone:
        flash(f"Invalid or missing phone number: {raw_phone}", "error")
        return redirect(url_for("order_detail", order_id=order_id))

    try:
        callback_url = f"{BASE_URL}{url_for('voice_entry', order_id=order_id)}"

        call = signalwire_client.calls.create(
            to=phone,
            from_=SIGNALWIRE_CALLER_ID,
            url=callback_url,
        )

        order["last_call_sid"] = call.sid
        flash("Call to customer has been initiated via SignalWire.", "success")
        return render_template("call_status.html", order=order, call_sid=call.sid)

    except Exception as e:
        print("SignalWire call error:", repr(e))
        flash(f"Failed to start call: {e}", "error")
        return redirect(url_for("order_detail", order_id=order_id))


# -------------------------------------------------
# SignalWire webhooks: voice flow (cXML)
# -------------------------------------------------
@app.route("/voice_entry/<int:order_id>", methods=["GET", "POST"])
def voice_entry(order_id):
    """
    Called by SignalWire when the outbound call is answered.
    We use <Gather> with speech input to ask for confirmation.
    """
    order = orders.get(order_id)
    if not order:
        vr = VoiceResponse()
        vr.say("দুঃখিত, অর্ডারটি খুঁজে পাওয়া যায়নি।", language="bn-BD")
        vr.hangup()
        return Response(str(vr), mimetype="text/xml")

    script = order["script"]

    vr = VoiceResponse()

    gather = Gather(
        action=f"{BASE_URL}{url_for('handle_reply', order_id=order_id)}",
        method="POST",
        input="speech",
        speechTimeout="auto",
        language="bn-BD",
        timeout=10,
    )

    gather.say(script, language="bn-BD")
    vr.append(gather)

    # If no speech was captured, fallback
    vr.say(
        "দুঃখিত, আপনার কাছ থেকে কোনো উত্তর পাওয়া যায়নি। পরে আবার চেষ্টা করা হবে।",
        language="bn-BD",
    )
    vr.hangup()

    return Response(str(vr), mimetype="text/xml")


@app.route("/handle_reply/<int:order_id>", methods=["GET", "POST"])
def handle_reply(order_id):
    """
    SignalWire posts SpeechResult / Digits here after Gather.
    We classify and update the order status.
    """
    order = orders.get(order_id)
    vr = VoiceResponse()

    if not order:
        vr.say("দুঃখিত, অর্ডারটি খুঁজে পাওয়া যায়নি।", language="bn-BD")
        vr.hangup()
        return Response(str(vr), mimetype="text/xml")

    speech = request.values.get("SpeechResult", "")
    digits = request.values.get("Digits")  # if you later support keypad input

    decision = classify_customer_reply(speech)
    order["last_result"] = {
        "speech": speech,
        "digits": digits,
        "decision": decision,
        "at": datetime.utcnow().isoformat(),
    }

    if decision == "confirmed":
        order["status"] = "confirmed"
        vr.say(
            "ধন্যবাদ। আপনার অর্ডার কনফার্ম করা হয়েছে। ইনশাআল্লাহ খুব দ্রুতই ডেলিভারি দেওয়া হবে।",
            language="bn-BD",
        )
    elif decision == "cancelled":
        order["status"] = "cancelled"
        vr.say(
            "আপনার অর্ডার বাতিল করা হয়েছে। ধন্যবাদ। ভবিষ্যতে আবার আমাদের সাথে থাকবেন ইনশাআল্লাহ।",
            language="bn-BD",
        )
    else:
        order["status"] = "needs_review"
        vr.say(
            "দুঃখিত, আপনার কথা ঠিকভাবে বোঝা যায়নি। আমাদের টিম থেকে একজন মানুষ আপনার সাথে যোগাযোগ করবে। ধন্যবাদ।",
            language="bn-BD",
        )

    vr.hangup()
    return Response(str(vr), mimetype="text/xml")



@app.route("/local_interact/<int:order_id>")
def local_interact(order_id):
    """
    Show a page where user can talk to the bot using laptop mic/speaker
    (no SignalWire, no real phone call).
    """
    order = orders.get(order_id)
    if not order:
        flash("Order not found.", "error")
        return redirect(url_for("index"))
    return render_template("local_interact.html", order=order)


@app.route("/api/interpret", methods=["POST"])
def api_interpret():
    """
    AJAX endpoint: browser sends recognized speech text,
    we classify intent and return a Bangla reply.
    """
    data = request.get_json(force=True)
    text = data.get("text", "") or ""
    decision = classify_customer_reply(text)

    if decision == "confirmed":
        reply = (
            "ধন্যবাদ। আপনার অর্ডার কনফার্ম করা হয়েছে। "
            "খুব শিগগিরই আমরা ডেলিভারি প্রসেস শুরু করব ইনশাআল্লাহ।"
        )
    elif decision == "cancelled":
        reply = (
            "আপনার অর্ডার বাতিল করা হয়েছে। "
            "ধন্যবাদ আমাদেরকে জানানোর জন্য। ভবিষ্যতে আবার আমাদের সাথে থাকবেন।"
        )
    else:
        reply = (
            "দুঃখিত, আপনার উত্তরটি পরিষ্কারভাবে বোঝা যায়নি। "
            "যদি কনফার্ম করতে চান, বলুন ‘হ্যাঁ, অর্ডার কনফার্ম’। "
            "বাতিল করতে চাইলে বলুন ‘না, অর্ডার ক্যান্সেল’।"
        )

    return jsonify({"decision": decision, "reply": reply})


@app.route("/local_bot")
def local_bot():
    """
    Standalone local voice bot page (no SignalWire, no phone, no order).
    """
    return render_template("local_bot.html")





def humanize_reply(text: str) -> str:
    """
    Make the Bangla reply sound a bit more like a real call-center agent.
    Adds light fillers and smoother pauses.
    """
    if not text:
        return text

    # Light fillers at the start (not always)
    fillers = [
        "আচ্ছা স্যার,",
        "জি স্যার,",
        "ঠিক আছে স্যার,",
        "হুম স্যার,"
    ]

    stripped = text.strip()
    if (random.random() < 0.3 and
        not stripped.startswith(("স্যার", "আচ্ছা", "জি", "ঠিক আছে"))):
        text = random.choice(fillers) + " " + stripped
    else:
        text = stripped


    # Make sentence breaks less abrupt
    text = text.replace("।  ", "। ")
    text = text.replace("। তারপর", "... তারপর")
    text = text.replace("। কিন্তু", "... কিন্তু")

    return text


def emotional_touch(text: str) -> str:
    """
    Add some polite/emotional tone to common phrases.
    """
    replacements = {
        "ঠিক আছে": "ঠিক আছে স্যার",
        "বুঝেছি": "জি স্যার, বুঝেছি",
        "ধন্যবাদ": "অনেক ধন্যবাদ স্যার",
    }
    for src, tgt in replacements.items():
        text = text.replace(src, tgt)
    return text

# Map Bangla digits → English digits for conversion
BENGALI_DIGIT_MAP = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

def normalize_numbers_for_bangla_tts(text: str) -> str:
    """
    Convert numeric sequences (120, ১২০, 1200) into Bangla words
    so TTS says 'একশ কুড়ি' instead of 'ওয়ান টু জিরো'.
    Only affects speech; original text reply stays unchanged.
    """
    def repl(match):
        raw = match.group(0)
        # Convert any Bangla digits to ASCII digits
        ascii_digits = raw.translate(BENGALI_DIGIT_MAP)
        try:
            n = int(ascii_digits)
        except ValueError:
            return raw

        try:
            words_bn = num2words(n, lang="bn")
        except Exception:
            return raw  # fallback: keep original

        return words_bn

    # Match both English and Bangla digit sequences
    return re.sub(r"[0-9০-৯]+", repl, text)




def postprocess_bot_text(text: str) -> str:
    """
    Combine humanization + emotional tone + light cleanup.
    """
    text = text.strip()
    text = emotional_touch(text)
    text = humanize_reply(text)
    return text

def synthesize_bangla_tts(text: str) -> str:
    """
    Generate Bangla TTS audio from given text and return the static URL.
    Shared by normal replies and the initial welcome message.
    """
    tts_dir = os.path.join("static", "tts")
    os.makedirs(tts_dir, exist_ok=True)
    filename = f"tts_{uuid.uuid4().hex}.wav"
    filepath = os.path.join(tts_dir, filename)

    # Light cleanup to avoid too many hard stops that cause long pauses
    cleaned = text

    # Convert specific patterns to commas for smoother flow
    cleaned = cleaned.replace("। তারপর", ", তারপর")

    # Convert digits (120, ১২০, 1200) to Bangla words for clear speaking
    cleaned = normalize_numbers_for_bangla_tts(cleaned)

    # 1) Generate raw Bangla TTS with cleaned text
    bn_tts.tts_to_file(text=cleaned, file_path=filepath)

    # 2) Normalize volume to a consistent loudness
    audio = AudioSegment.from_file(filepath, format="wav")
    target_dBFS = -16.0  # typical clear-voice loudness
    change_in_dBFS = target_dBFS - audio.dBFS
    normalized_audio = audio.apply_gain(change_in_dBFS)

    # 3) Optional: light fade-in / fade-out to avoid clicks
    normalized_audio = normalized_audio.fade_in(20).fade_out(50)

    # 4) Overwrite the file with normalized audio
    normalized_audio.export(filepath, format="wav")

    return url_for("static", filename=f"tts/{filename}")

@app.route("/api/local_bot_welcome", methods=["GET"])
def api_local_bot_welcome():
    """
    First sentence from the bot when user clicks Start in Local Voice Bot.
    """
    intro = (
        "আসসালামু আলাইকুম। আমি একজন বট কথা বলছি। "
        "আপনি একটি শার্ট অর্ডার করেছেন। "
        "অনুগ্রহ করে শার্টের মডেল, রঙ আর সাইজ বলুন। "
        "অর্ডার ঠিক থাকলে বলবেন – ‘হ্যাঁ, অর্ডার কনফার্ম’। "
        "বাতিল করতে চাইলে বলবেন – ‘না, অর্ডার ক্যান্সেল’।"
    )

    try:
        audio_url = synthesize_bangla_tts(intro)
    except Exception as e:
        print("Coqui TTS error in welcome:", repr(e))
        return jsonify({
            "reply": intro,
            "audio_url": None,
            "error": f"Coqui TTS error: {e}",
        }), 500

    return jsonify({"reply": intro, "audio_url": audio_url})



@app.route("/api/local_bot", methods=["POST"])
def api_local_bot():
    """
    Browser sends full conversation; we call LLaMA to generate the next reply,
    then synthesize Bangla audio using local Coqui Bangla TTS.
    """
    data = request.get_json(force=True)
    messages = data.get("messages", [])

    if not isinstance(messages, list):
        return jsonify({"error": "messages must be a list"}), 400

    system_prompt = (
        "তুমি একজন বাংলাদেশের কল সেন্টার এজেন্ট, কাজ শুধু শার্ট অর্ডার কনফার্ম করা। "
        "সব সময় শুধু বাংলা ভাষায় কথা বলবে। "
        "তুমি শুধু এই বিষয়গুলো নিয়ে কথা বলতে পারো: শার্টের সংখ্যা, কালার, সাইজ, দাম, "
        "কাস্টমারের নাম, মোবাইল নাম্বার, ডেলিভারি অ্যাড্রেস, অর্ডার কনফার্ম/ক্যান্সেল। "
        "এর বাইরে কোনো টপিক, সাধারণ কথা, পরামর্শ, মজার কথা, জ্ঞানগর্ভ কথা কিছুই বলবে না। "
        "যদি ইউজার অন্য কিছু জিজ্ঞেস করে বা অন্য বিষয়ে চলে যায়, তুমি সংক্ষিপ্তভাবে এভাবে বলবে: "
        "“স্যার, আমি শুধু আপনার শার্ট অর্ডার কনফার্ম করার জন্য আছি, "
        "অনুগ্রহ করে অর্ডারের তথ্য বলুন।” "
        "একবার উত্তরে সর্বোচ্চ ১–২টি ছোট বাক্য ব্যবহার করবে, "
        "ভদ্র, পরিষ্কার এবং সহজ ভাষায় কথা বলবে। "
    )

    groq_messages = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if not content:
            continue
        if role not in {"user", "assistant"}:
            role = "user"
        groq_messages.append({"role": role, "content": content})

    # 1) Get Bangla reply text from LLaMA
    try:
        chat = groq_client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=groq_messages,
            temperature=0.2,
        )
        reply = chat.choices[0].message.content
        reply = postprocess_bot_text(reply)

    except Exception as e:
        print("Groq error in /api_local_bot:", repr(e))
        return jsonify({
            "reply": "",
            "audio_url": None,
            "error": f"Groq error: {e}"
        }), 500

    # 2) Generate Bangla speech using shared helper
    try:
        audio_url = synthesize_bangla_tts(reply)
    except Exception as e:
        print("Coqui Bangla TTS error:", repr(e))
        return jsonify({
            "reply": reply,
            "audio_url": None,
            "error": f"Coqui TTS error: {e}"
        }), 500

    return jsonify({"reply": reply, "audio_url": audio_url})



# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
