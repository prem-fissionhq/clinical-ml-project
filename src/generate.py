import pandas as pd
import numpy as np
np.random.seed(42)
n = 1000
df = pd.DataFrame({
"age": np.random.randint(25, 85, n),
"bmi": np.round(np.random.normal(27, 5, n), 1),
"blood_pressure": np.random.randint(90, 180, n),
"cholesterol": np.random.randint(150, 300, n),
"treatment_group": np.random.choice(["A", "B", "placebo"], n),
"outcome": np.random.choice([0, 1], n, p=[0.6, 0.4]),
})
df.to_csv("data/patient_outcomes.csv", index=False)
print(f"Generated {len(df)} patient records")