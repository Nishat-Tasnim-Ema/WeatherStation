"""
Flask API — serves rain / temp / humidity predictions from the models
trained in Colab on hourly Dhaka data.

IMPORTANT: the ESP32 pushes every 5 minutes, but the model was trained on
HOURLY rows (temp_roll_3h = mean of last 3 hourly readings, HORIZON = 6
hours). Feeding it raw 5-minute rows would silently break the features.
This version resamples the live feed to hourly before building features.
"""

from flask import Flask, jsonify
import joblib
import requests
import pandas as pd
import os
import threading
from telegram.ext import ApplicationBuilder, CommandHandler

app = Flask(__name__)

# Never hardcode this — set BOT_TOKEN in Render's Environment tab.
BOT_TOKEN = os.environ.get("BOT_TOKEN")

rain_clf = joblib.load("rain_model.pkl")
temp_reg = joblib.load("temp_model.pkl")
hum_reg = joblib.load("humidity_model.pkl")

# Same feature order used at training time (no rain_analog — that field
# doesn't exist in the historical CSV the models were trained on).
FEATURES = [
    "temperature", "humidity", "hour", "month",
    "temp_roll_3h", "humidity_roll_3h", "temp_trend", "humidity_trend", "rain_recent",
]

FIREBASE_URL = "https://weather-station-91545-default-rtdb.asia-southeast1.firebasedatabase.app"

# 5-min pushes -> need at least 4-5 hours of raw rows to get 3+ clean hourly
# buckets after resampling. limitToLast=70 gives ~5.8 hours of headroom.
RAW_ROWS_TO_FETCH = 70


def fetch_hourly_features():
    """Pull recent raw pushes from Firebase, resample to hourly, and build
    the same engineered features used during training."""
    url = f'{FIREBASE_URL}/sensor_data.json?orderBy="timestamp"&limitToLast={RAW_ROWS_TO_FETCH}'
    data = requests.get(url, timeout=10).json()
    if not data:
        raise ValueError("No data returned from Firebase")

    raw = pd.DataFrame.from_dict(data, orient="index")
    raw["time"] = pd.to_datetime(raw["timestamp"], unit="s")
    raw = raw.sort_values("time").set_index("time")

    # --- Resample 5-min pushes to hourly, matching training resolution ---
    hourly = pd.DataFrame()
    hourly["temperature"] = raw["temperature"].resample("1h").mean()
    hourly["humidity"] = raw["humidity"].resample("1h").mean()
    # rain_digital: did it rain AT ALL during that hour, not the average
    hourly["rain_digital"] = raw["rain_digital"].resample("1h").max()
    hourly = hourly.dropna(subset=["temperature", "humidity"])

    if len(hourly) < 2:
        raise ValueError(
            f"Only {len(hourly)} complete hourly bucket(s) available yet — "
            "need at least ~3 hours of pushes before predictions are meaningful."
        )

    # --- Same engineered features as training (window=3 -> 3 hourly rows) ---
    hourly["temp_roll_3h"] = hourly["temperature"].rolling(window=3, min_periods=1).mean()
    hourly["humidity_roll_3h"] = hourly["humidity"].rolling(window=3, min_periods=1).mean()
    hourly["temp_trend"] = hourly["temperature"].diff().fillna(0)
    hourly["humidity_trend"] = hourly["humidity"].diff().fillna(0)
    hourly["rain_recent"] = hourly["rain_digital"].rolling(window=3, min_periods=1).max()

    last = hourly.iloc[-1]
    last_ts = hourly.index[-1]

    row = pd.DataFrame([{
        "temperature": last["temperature"],
        "humidity": last["humidity"],
        "hour": last_ts.hour,
        "month": last_ts.month,
        "temp_roll_3h": last["temp_roll_3h"],
        "humidity_roll_3h": last["humidity_roll_3h"],
        "temp_trend": last["temp_trend"],
        "humidity_trend": last["humidity_trend"],
        "rain_recent": last["rain_recent"],
    }])[FEATURES]

    # also return the most recent raw reading for "current" display values
    latest_raw = raw.iloc[-1]
    return row, latest_raw, last_ts


@app.route("/predict")
def predict():
    try:
        row, latest_raw, bucket_ts = fetch_hourly_features()
    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 503

    rain_prob = rain_clf.predict_proba(row)[0][1]

    return jsonify({
        "ok": True,
        "current_temp": round(float(latest_raw["temperature"]), 1),
        "current_humidity": round(float(latest_raw["humidity"]), 1),
        "current_rain_analog": int(latest_raw["rain_analog"]),
        "feature_bucket_hour": bucket_ts.strftime("%Y-%m-%d %H:00"),
        "predicted_rain_chance_6h": round(float(rain_prob) * 100, 1),
        "predicted_temp_6h": round(float(temp_reg.predict(row)[0]), 1),
        "predicted_humidity_6h": round(float(hum_reg.predict(row)[0]), 1),
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# --- Telegram bot -----------------------------------------------------
LOCAL_PORT = os.environ.get("PORT", "5000")


async def weather_command(update, context):
    r = requests.get(f"http://localhost:{LOCAL_PORT}/predict").json()
    if not r.get("ok"):
        await update.message.reply_text(f"⚠️ {r.get('error', 'prediction not ready yet')}")
        return
    msg = (
        f"🌡 Current: {r['current_temp']}°C, {r['current_humidity']}% humidity\n"
        f"🌧 Rain chance (6h): {r['predicted_rain_chance_6h']}%\n"
        f"📈 Forecast: {r['predicted_temp_6h']}°C, {r['predicted_humidity_6h']}% humidity"
    )
    await update.message.reply_text(msg)


def run_bot():
    if not BOT_TOKEN:
        print("BOT_TOKEN not set — skipping Telegram bot startup.")
        return
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    application = ApplicationBuilder().token(BOT_TOKEN).build()
    application.add_handler(CommandHandler("weather", weather_command))
    # stop_signals=None: signal handlers only work in the main thread, and
    # this runs in a background thread, so skip registering them.
    application.run_polling(stop_signals=None)


# Gunicorn imports this module rather than running it as __main__, so the
# bot thread has to start at import time (not inside `if __name__ == ...`)
# or it will never launch in production on Render.
threading.Thread(target=run_bot, daemon=True).start()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(LOCAL_PORT))