# 🚢 Titanic Survival Predictor

ระบบทำนายการรอดชีวิตของผู้โดยสารเรือไททานิก ด้วยโมเดล SVM (Support Vector Machine)
พัฒนาโดย **นายศักดิโชติ แตงโสภา**

## ไฟล์ในโปรเจกต์
- `app.py` — โค้ดหลักของแอป Streamlit
- `titanic_svm.joblib` — โมเดล SVM ที่เทรนไว้แล้ว
- `requirements.txt` — รายการไลบรารีที่ต้องติดตั้ง

## ⚠️ สิ่งที่ควรตรวจสอบก่อนใช้งานจริง
โมเดลไฟล์ที่แนบมาไม่มีข้อมูลชื่อ feature หรือ scaler แนบมาด้วย ผมจึงอนุมาน
ลำดับ feature และค่าที่ใช้ scale จากลักษณะของข้อมูลในโมเดล (ดูคอมเมนต์ใน
`app.py` ส่วน `SCALER_PARAMS` และ `FEATURE_ORDER`) โดยสันนิษฐานว่า:

```
ลำดับ feature: [Pclass, Sex, Age, Fare, SibSp]
```

**ถ้าคุณทราบลำดับ/ค่า scaler ที่ใช้เทรนจริง** กรุณาแก้ไขค่าในส่วนนั้นของ
`app.py` ให้ตรงกับของจริง เพื่อให้ผลการทำนายแม่นยำ และแก้ค่า `MODEL_ACCURACY`
เป็นค่าความแม่นยำจริงที่ได้จากการประเมินผลบนชุดทดสอบของคุณ

## 🖥️ รันบนเครื่องตัวเอง (Local)
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 📤 อัปโหลดขึ้น GitHub
```bash
git init
git add app.py requirements.txt titanic_svm.joblib README.md
git commit -m "Titanic survival predictor app"
git branch -M main
git remote add origin https://github.com/<ชื่อผู้ใช้>/<ชื่อ repo>.git
git push -u origin main
```

## ☁️ Deploy บน Streamlit Community Cloud
1. เข้า https://share.streamlit.io/ แล้วล็อกอินด้วย GitHub
2. กด **New app** แล้วเลือก repository ที่เพิ่ง push ไป
3. ตั้งค่า **Main file path** เป็น `app.py`
4. กด **Deploy** รอสักครู่ แอปจะได้ลิงก์สาธารณะให้ใช้งาน
