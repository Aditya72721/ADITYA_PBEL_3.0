# Ye hamari website ka "backend" hai — matlab jo cheez screen
# ke peeche chal rahi hai. Iska kaam simple hai:
#   1. Website ka page dikhana (home page)
#   2. User jo transaction detail bhare, usko model ko dikhana
#   3. Model se answer (fraud % kitna hai) leke wapas bhejna
#   4. Ek "random sample" bhi de dena taaki user demo try kar sake

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import random
import time

app = Flask(__name__)  # apni website ko "app" naam se start kar rahe hain

# server start hote hi ek hi baar model load kar lo (baar baar
# load karna slow hota, isliye upar hi kar diya)
bundle = joblib.load("model.pkl")
model = bundle["model"]        # asli trained model
FEATURES = bundle["features"]  # model ko kaunse columns chahiye, uski list

# ROUTE 1: Home page
# jab koi browser mein "/" (yani localhost:5000) kholega, to
# ye function chalega aur index.html dikha dega
@app.route("/")
def home():
    return render_template("index.html")
# ROUTE 2: asli fraud check yahi hota hai
# jab user form fill karke "Scan transaction" dabata hai, browser
# se ye data yahan POST request ke through aata hai
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()  # frontend se jo JSON aaya, usko utha lo

    try:
        # amount_ratio ko chhod ke baaki sab values seedhe form se le lo
        base = {f: float(data[f]) for f in FEATURES if f != "amount_ratio"}

        # ye wahi "smart trick" hai jo humne train_model.py mein bhi
        # use ki thi — current amount ko user ke normal average se
        # compare karo. Isse chhoti transactions galat flag nahi hoti.
        base["amount_ratio"] = base["amount"] / (base["avg_amount_30d"] + 1)

        # model ko exactly wahi order mein values chahiye jisme usne
        # training ke time dekhi thi, isliye FEATURES list follow kar rahe hain
        row = [base[f] for f in FEATURES]
    except (KeyError, ValueError, TypeError):
        # agar user ne kuch galat bhej diya (jaise text field mein letter)
        # to crash hone ke bajaye clean error bhej do
        return jsonify({"error": "Invalid input"}), 400

    # yahi asli prediction ho rahi hai — model bata raha hai fraud
    # hone ka probability (0 se 1 ke beech, isliye *100 karke % bana diya)
    proba = model.predict_proba([row])[0][1]
    risk_score = round(proba * 100, 1)

    # risk score ke hisaab se ek insaan-samajhne-wala label de rahe hain
    # (ye thresholds tune kiye gaye hain taaki chhoti/normal
    # transactions FRAUD na ban jaye)
    if risk_score >= 80:
        verdict = "FRAUD"
        label = "High risk — flag for review"
    elif risk_score >= 50:
        verdict = "SUSPICIOUS"
        label = "Medium risk — extra verification suggested"
    else:
        verdict = "LEGIT"
        label = "Low risk — looks clean"

    # ye sirf dikhawe ke liye hai — thoda sa delay taaki frontend ka
    # "scanning" animation real jaisa lage, warna itni fast reply
    # aati hai ki animation dikhega hi nahi
    time.sleep(0.6)

    return jsonify({
        "risk_score": risk_score,
        "verdict": verdict,
        "label": label,
    })

# ROUTE 3: "Try random sample" button ke liye
# ye ek dummy transaction bana ke de deta hai (kabhi fraud jaisi,
# kabhi legit jaisi) taaki user bina khud data type kiye demo dekh sake
@app.route("/sample")
def sample():
    """Returns a random plausible transaction, used by the 'Try a sample' button."""

    # 50-50 chance rakha hai fraud-jaisi ya legit-jaisi transaction banane ka
    is_fraud_like = random.random() < 0.5

    if is_fraud_like:
        # fraud-jaisa pattern: bada amount, raat ka time, bahut saari
        # transactions, foreign, naya merchant — bilkul wahi red flags
        # jo humne model train karte waqt use kiye the
        txn = {
            "amount": round(random.uniform(400, 2200), 2),
            "hour_of_day": random.choice([1, 2, 3, 4, 23]),
            "txn_count_24h": random.randint(6, 14),
            "avg_amount_30d": round(random.uniform(50, 300), 2),
            "is_foreign": 1,
            "is_weekend": random.choice([0, 1]),
            "new_merchant": 1,
        }
    else:
        # normal/legit pattern: chhota amount, din ka time, kam
        # transactions, apna hi desh
        txn = {
            "amount": round(random.uniform(8, 250), 2),
            "hour_of_day": random.randint(9, 21),
            "txn_count_24h": random.randint(0, 4),
            "avg_amount_30d": round(random.uniform(50, 300), 2),
            "is_foreign": 0,
            "is_weekend": random.choice([0, 1]),
            "new_merchant": random.choice([0, 1]),
        }
    return jsonify(txn)



# yahan se website actually start hoti hai
if __name__ == "__main__":
    # debug=True rakha hai taaki code change karte hi server khud
    # restart ho jaye — final/production mein ise False rakhna chahiye
    app.run(debug=True, host="0.0.0.0", port=5000)
