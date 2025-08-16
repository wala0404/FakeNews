import os, pandas as pd
os.makedirs("data/processed", exist_ok=True)

rows = [
    {"text": "Official statement confirms new policy", "label": 1},
    {"text": "Breaking!!! Aliens landed in Gaza", "label": 0},
    {"text": "وزارة الصحة تصدر تقريرًا رسميًا عن اللقاحات", "label": 1},
    {"text": "عاجل: شرب الماء الساخن يذيب الفيروسات فورًا", "label": 0},
    {"text": "Tech company releases earnings report", "label": 1},
    {"text": "صورة قديمة تُظهر حدثًا كأنه اليوم", "label": 0},
    {"text": "الرئاسة تعلن عن موعد الانتخابات", "label": 1},
    {"text": "دواء سحري يشفي كل الأمراض بيوم", "label": 0},
]


df = pd.DataFrame(rows)
df.iloc[:6].to_csv("data/processed/train.csv", index=False)
df.iloc[6:7].to_csv("data/processed/val.csv", index=False)
df.iloc[7:].to_csv("data/processed/test.csv", index=False)
print("Dummy data written to data/processed/")
