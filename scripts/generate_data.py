import pandas as pd
import numpy as np
import os
os.makedirs("data", exist_ok=True)
np.random.seed(42)
n = 1000
data = {
    "StudentID": [f"S{str(i).zfill(4)}" for i in range(1, n+1)],
    "GPA": np.round(np.random.uniform(1.5, 4.0, n), 2),
    "Attendance": np.random.randint(30, 100, n),
    "Fee_Status": np.random.choice([0, 1], n, p=[0.7, 0.3]),
    "Semester": np.random.choice([1,2,3,4,5,6,7,8], n),
}
df = pd.DataFrame(data)
df["Target"] = 0
df.loc[(df["GPA"] < 2.5) & (df["Attendance"] < 60), "Target"] = 1
df.loc[(df["Fee_Status"] == 1) & (df["GPA"] < 3.0), "Target"] = 1
df.to_csv("data/student_data.csv", index=False)
print("✅ Data created!")
