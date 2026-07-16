

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# seed fix kar diya taaki jab bhi ye code chalao, same random
# data bane — warna har baar different result aayega aur confuse
# ho jaoge ki model sahi hai ya nahi
np.random.seed(42)

# total kitne transactions ka data banana hai — jitna zyada utna
# better model banega, but training time bhi thoda badhega
N = 30000
# STEP 1: LEGIT (normal, sahi) transactions banao
# real life mein zyadatar transactions genuine hote hain, fraud
# bahut kam hota hai (jaise 3-4%). isliye humne 96% legit rakha.
n_legit = int(N * 0.96)

legit = pd.DataFrame({
    # amount ka average roughly ₹140 ke aas paas rakha (gamma
    # distribution use kiya kyunki real spending aisi hi dikhti
    # hai — zyada log kam amount spend karte hain, kabhi kabhi
    # bada amount bhi)
    "amount": np.random.gamma(2, 70, n_legit),

    # normal log din mein transaction karte hain, isliye peak
    # dopahar 2 baje (14:00) ke aas paas rakha
    "hour_of_day": np.random.normal(14, 5, n_legit).clip(0, 23),

    # ek normal banda din mein 3 transaction karta hai average
    "txn_count_24h": np.random.poisson(3, n_legit),

    # pichle 30 din ka average spending
    "avg_amount_30d": np.random.gamma(2, 65, n_legit),

    # kuch log travel bhi karte hain / videsh se shopping karte
    # hain, to thoda percentage foreign transaction bhi legit
    # logo ka rakha (14%) — warna model sochega "foreign = fraud"
    # jo galat hai
    "is_foreign": np.random.binomial(1, 0.14, n_legit),

    "is_weekend": np.random.binomial(1, 0.28, n_legit),

    # naya shop try karna bhi normal hai, isliye 35% legit logo
    # ne bhi new merchant use kiya hai
    "new_merchant": np.random.binomial(1, 0.35, n_legit),
})
legit["is_fraud"] = 0  # 0 matlab fraud nahi hai

# STEP 2: FRAUD transactions banao (jaan-bujh kar "risky" pattern)

n_fraud = N - n_legit

fraud = pd.DataFrame({
    # fraud transactions mein amount usually zyada hota hai
    "amount": np.random.gamma(2.5, 160, n_fraud),

    # fraud raat ko / late night zyada hota hai (log so rahe
    # hote hain, card cheezein cheezein chura ke use karte hain)
    "hour_of_day": np.random.normal(4, 4, n_fraud).clip(0, 23),

    # fraud karne wale ek hi din mein bahut saari transaction
    # try karte hain (card test karna / jaldi paisa nikalna)
    "txn_count_24h": np.random.poisson(7, n_fraud),

    "avg_amount_30d": np.random.gamma(2, 65, n_fraud),

    # fraud mein foreign transaction ka chance zyada hota hai
    "is_foreign": np.random.binomial(1, 0.45, n_fraud),

    "is_weekend": np.random.binomial(1, 0.33, n_fraud),

    # fraud waale zyadatar naye/unknown merchant pe hi transaction
    # karte hain
    "new_merchant": np.random.binomial(1, 0.55, n_fraud),
})
fraud["is_fraud"] = 1  # 1 matlab ye fraud hai

# ab dono ko mila do aur shuffle kar do (warna sara fraud neeche
# hi rahega, model ko confuse kar dega)
df = pd.concat([legit, fraud], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
# STEP 3: sabse IMPORTANT trick -> "amount_ratio" feature
# problem ye thi ki pehle model sirf "foreign" ya "new merchant"
# dekh ke hi transaction ko suspicious bol deta tha, chahe amount
# chhota hi kyu na ho. Isliye ye naya column banaya:
#
# amount_ratio = is transaction ka amount / tumhara normal average
#
# matlab agar tum roz ₹1500 ke aas paas spend karte ho aur aaj bhi
# ₹1500 ka transaction hua, to ratio ~1 hoga (bilkul normal).
# Lekin agar achanak ₹50,000 ka transaction aa gaya jab tum
# normally ₹500 hi spend karte ho, to ratio bahut zyada hoga —
# YE hai asli red flag, sirf "foreign" hona nahi.
df["amount_ratio"] = df["amount"] / (df["avg_amount_30d"] + 1)  # +1 taaki 0 se divide na ho

# model ko kaunse columns dikhane hain, wo list bana li
FEATURES = ["amount", "hour_of_day", "txn_count_24h", "avg_amount_30d",
            "amount_ratio", "is_foreign", "is_weekend", "new_merchant"]

X = df[FEATURES]        # inputs (jo model ko dikhega)
y = df["is_fraud"]      # output (jo model ko predict karna hai)


# STEP 4: data ko train aur test mein baant do
# 80% data se model seekhega (train), 20% pe test karenge ki
# model ne sahi seekha ya nahi (jo data usne dekha hi nahi)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# STEP 5: model banao aur train karo
# RandomForest matlab bahut saare chote-chote decision trees
# (300 yahan pe) jo milke vote karte hain "fraud hai ya nahi" —
# akela ek tree galti kare to baaki sab uska balance kar dete hain
model = RandomForestClassifier(
    n_estimators=300,        # kitne trees banayenge
    max_depth=8,             # ek tree kitna "deep" soch sakta hai (zyada depth = overfitting ka risk)
    min_samples_leaf=15,     # bahut chote/random patterns pe fasna mat, thoda generalize karo
    class_weight="balanced", # fraud data kam hai isliye usko zyada importance do
    random_state=42
)
model.fit(X_train, y_train)  # yahi asli "training" ho rahi hai


# STEP 6: model ko test karke dekho kitna accha bana

y_pred = model.predict(X_test)               # final answer (fraud/legit)
y_proba = model.predict_proba(X_test)[:, 1]  # kitna % confidence hai fraud hone ka

print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))
print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 4))
# ROC-AUC jitna 1.0 ke paas utna better model — 0.98-0.99 kaafi tagra maana jata hai


# STEP 7: sanity check -> chhoti/normal transaction galat se
# fraud na bole, isko manually verify kar rahe hain

sanity = pd.DataFrame([
    {"amount": 500, "hour_of_day": 14, "txn_count_24h": 1, "avg_amount_30d": 600,
     "is_foreign": 0, "is_weekend": 0, "new_merchant": 1},
    {"amount": 1500, "hour_of_day": 10, "txn_count_24h": 1, "avg_amount_30d": 1650,
     "is_foreign": 1, "is_weekend": 0, "new_merchant": 1},
])
sanity["amount_ratio"] = sanity["amount"] / (sanity["avg_amount_30d"] + 1)
print("\nSanity check (chhoti/normal transactions ka score dekho, low hona chahiye):")
print(model.predict_proba(sanity[FEATURES])[:, 1])

# STEP 8: trained model ko file mein save kar do
# taaki har baar dubara train na karna pade — Flask app seedha
# ye file uthake use kar lega
joblib.dump({"model": model, "features": FEATURES}, "model.pkl")
print("\nSaved model.pkl — ab ye file app.py use karega")
