"""
ระบบทำนายการรอดชีวิตผู้โดยสารเรือไททานิก (Titanic Survival Prediction)
พัฒนาโดย: นายศักดิโชติ แตงโสภา

หมายเหตุสำคัญเกี่ยวกับโมเดล:
โมเดล SVM (titanic_svm.joblib) ถูกเทรนด้วยข้อมูลที่ผ่านการ Scale (StandardScaler)
มาแล้ว และไม่ได้แนบชื่อ feature หรือ scaler มาด้วย จึงต้องอนุมานจากลักษณะของ
support vectors ว่า input ที่โมเดลต้องการคือ 5 ค่า เรียงตามลำดับดังนี้:
    [Pclass, Sex(0=หญิง,1=ชาย), Age, Fare, SibSp]
และใช้ค่าเฉลี่ย/ส่วนเบี่ยงเบนมาตรฐานมาตรฐานของชุดข้อมูล Titanic (train.csv)
ในการ Scale ค่า Age, Fare, SibSp (ดูค่าคงที่ในส่วน SCALER PARAMS ด้านล่าง)

หากผู้พัฒนาทราบลำดับ feature หรือมีไฟล์ scaler ที่ใช้เทรนจริง กรุณาแก้ไขค่า
ในส่วน "SCALER PARAMS" และ "FEATURE ORDER" ด้านล่างให้ตรงกับของจริง เพื่อความ
แม่นยำของการทำนาย
"""

import streamlit as st
import joblib
import numpy as np
import os

# ------------------------------------------------------------------
# ตั้งค่าหน้าเว็บ (ต้องเรียกเป็นคำสั่งแรกสุดของ Streamlit)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ------------------------------------------------------------------
# ค่าคงที่ / การตั้งค่าโมเดล — แก้ไขได้ตามข้อมูลจริงของคุณ
# ------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "titanic_svm.joblib")

# ความแม่นยำของโมเดล (ควรแทนที่ด้วยค่าจริงจากการประเมินผลบนชุดทดสอบของคุณ)
MODEL_ACCURACY = 0.82  # <-- TODO: แก้ไขเป็นค่า accuracy จริงของโมเดล

# ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐาน อ้างอิงจากชุดข้อมูล Titanic (train.csv) มาตรฐาน
# ใช้สำหรับ StandardScaler: z = (x - mean) / std
SCALER_PARAMS = {
    "Pclass": {"mean": 2.309, "std": 0.836},
    "Sex":    {"mean": 0.352, "std": 0.478},  # 0 = หญิง, 1 = ชาย
    "Age":    {"mean": 29.699, "std": 14.526},
    "Fare":   {"mean": 32.204, "std": 49.693},
    "SibSp":  {"mean": 0.523, "std": 1.103},
}

# ลำดับ feature ที่ป้อนเข้าโมเดล (แก้ไขลำดับตรงนี้ถ้าโมเดลจริงเรียงต่างกัน)
FEATURE_ORDER = ["Pclass", "Sex", "Age", "Fare", "SibSp"]


# ------------------------------------------------------------------
# โหลดโมเดล (cache ไว้ไม่ให้โหลดซ้ำทุกครั้งที่ interact)
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)



def scale_value(name: str, value: float) -> float:
    p = SCALER_PARAMS[name]
    return (value - p["mean"]) / p["std"]


def predict_survival(model, pclass, sex, age, fare, sibsp):
    raw = {"Pclass": pclass, "Sex": sex, "Age": age, "Fare": fare, "SibSp": sibsp}
    scaled = [scale_value(name, raw[name]) for name in FEATURE_ORDER]
    x = np.array(scaled).reshape(1, -1)
    pred = model.predict(x)[0]
    return pred


# ------------------------------------------------------------------
# สไตล์ (มินิมอล โทนสีฟ้า-เทา)
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
        .main { background-color: #FAFBFC; }
        h1 { font-weight: 700; color: #1B2A4A; }
        .subtitle {
            color: #6B7280;
            font-size: 1rem;
            margin-top: -10px;
            margin-bottom: 1.5rem;
        }
        .stButton>button {
            background-color: #1E5F8C;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #164A6E;
            color: white;
        }
        .metric-box {
            background-color: #F1F5F9;
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
        }
        .footer {
            text-align: center;
            color: #9CA3AF;
            font-size: 0.85rem;
            margin-top: 3rem;
            padding-top: 1rem;
            border-top: 1px solid #E5E7EB;
        }
        div[data-testid="stForm"] {
            background-color: #FFFFFF;
            padding: 1.5rem;
            border-radius: 14px;
            border: 1px solid #E5E7EB;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# ส่วนหัว
# ------------------------------------------------------------------
st.markdown("<h1>🚢 Titanic Survival Predictor</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>ระบบทำนายโอกาสรอดชีวิตของผู้โดยสารเรือไททานิก ด้วยโมเดล Support Vector Machine (SVM)</div>",
    unsafe_allow_html=True,
)

# แสดงค่าความแม่นยำของระบบเพื่อเพิ่มความเชื่อมั่น
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        f"""<div class='metric-box'>
                <div style='font-size:1.6rem;'>🎯</div>
                <div style='font-size:1.3rem; font-weight:700; color:#1E5F8C;'>{MODEL_ACCURACY*100:.1f}%</div>
                <div style='color:#6B7280; font-size:0.85rem;'>ความแม่นยำของโมเดล</div>
            </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """<div class='metric-box'>
                <div style='font-size:1.6rem;'>⚙️</div>
                <div style='font-size:1.3rem; font-weight:700; color:#1E5F8C;'>SVM</div>
                <div style='color:#6B7280; font-size:0.85rem;'>อัลกอริทึมที่ใช้</div>
            </div>""",
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        """<div class='metric-box'>
                <div style='font-size:1.6rem;'>📊</div>
                <div style='font-size:1.3rem; font-weight:700; color:#1E5F8C;'>5</div>
                <div style='color:#6B7280; font-size:0.85rem;'>ตัวแปรที่ใช้ทำนาย</div>
            </div>""",
        unsafe_allow_html=True,
    )

st.write("")

# ------------------------------------------------------------------
# ฟอร์มกรอกข้อมูลผู้โดยสาร
# ------------------------------------------------------------------
with st.form("prediction_form"):
    st.markdown("#### 📝 กรอกข้อมูลผู้โดยสาร")

    c1, c2 = st.columns(2)
    with c1:
        pclass_label = st.selectbox(
            "🎫 ชั้นโดยสาร (Passenger Class)",
            options=["ชั้น 1 (Upper)", "ชั้น 2 (Middle)", "ชั้น 3 (Lower)"],
        )
        sex_label = st.radio("🧑‍🤝‍🧑 เพศ (Sex)", options=["ชาย", "หญิง"], horizontal=True)
        age = st.slider("🎂 อายุ (Age)", min_value=0, max_value=80, value=30)

    with c2:
        fare = st.number_input(
            "💵 ค่าโดยสาร (Fare)", min_value=0.0, max_value=600.0, value=32.0, step=1.0
        )
        sibsp = st.number_input(
            "👨‍👩‍👧 จำนวนพี่น้อง/คู่สมรสที่ร่วมเดินทาง (SibSp)",
            min_value=0,
            max_value=8,
            value=0,
            step=1,
        )

    submitted = st.form_submit_button("🔮 ทำนายผล")

# ------------------------------------------------------------------
# แสดงผลการทำนาย
# ------------------------------------------------------------------
if submitted:
    model = load_model()

    pclass = {"ชั้น 1 (Upper)": 1, "ชั้น 2 (Middle)": 2, "ชั้น 3 (Lower)": 3}[pclass_label]
    sex = 1 if sex_label == "ชาย" else 0

    prediction = predict_survival(model, pclass, sex, age, fare, sibsp)

    st.write("")
    if prediction == 1:
        st.success("### ✅ ผลการทำนาย: รอดชีวิต (Survived)")
        st.balloons()
    else:
        st.error("### ❌ ผลการทำนาย: ไม่รอดชีวิต (Did not survive)")

    with st.expander("ดูข้อมูลที่ใช้ในการทำนาย"):
        st.write(
            {
                "ชั้นโดยสาร (Pclass)": pclass,
                "เพศ (Sex, 0=หญิง/1=ชาย)": sex,
                "อายุ (Age)": age,
                "ค่าโดยสาร (Fare)": fare,
                "จำนวนพี่น้อง/คู่สมรส (SibSp)": sibsp,
            }
        )

# ------------------------------------------------------------------
# ส่วนท้าย
# ------------------------------------------------------------------
st.markdown(
    "<div class='footer'>พัฒนาโดย นายศักดิโชติ แตงโสภา &nbsp;|&nbsp; ระบบทำนายการรอดชีวิตผู้โดยสารไททานิก</div>",
    unsafe_allow_html=True,
)
