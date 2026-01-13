import pandas as pd
import numpy as np

# List of major fruits and vegetables with some realistic shelf life baselines (in days)
produce_types = [
    {'Type': 'Banana', 'base_life': 14, 'opt_temp': 13, 'opt_hum': 85},
    {'Type': 'Apple', 'base_life': 60, 'opt_temp': 4, 'opt_hum': 90},
    {'Type': 'Tomato', 'base_life': 14, 'opt_temp': 12, 'opt_hum': 90},
    {'Type': 'Mango', 'base_life': 10, 'opt_temp': 13, 'opt_hum': 85},
    {'Type': 'Grape', 'base_life': 30, 'opt_temp': 0, 'opt_hum': 90},
    {'Type': 'Strawberry', 'base_life': 7, 'opt_temp': 0, 'opt_hum': 90},
    {'Type': 'Cucumber', 'base_life': 10, 'opt_temp': 10, 'opt_hum': 95},
    {'Type': 'Carrot', 'base_life': 30, 'opt_temp': 0, 'opt_hum': 95},
    {'Type': 'Potato', 'base_life': 90, 'opt_temp': 7, 'opt_hum': 90},
    {'Type': 'Onion', 'base_life': 90, 'opt_temp': 4, 'opt_hum': 70}
]


def shelf_life_decay(typeinfo, temp, hum):
    # Empirical shelf-life decay based on deviations from optimal temp/humidity:
    #  - higher temp/humidity reduces shelf life
    #  - severe deviation is strongly penalized
    base = typeinfo['base_life']
    t_opt = typeinfo['opt_temp']
    h_opt = typeinfo['opt_hum']
    # Penalize both higher and much lower than optimal temperature/humidity
    t_penalty = 1 + 0.07 * abs(temp - t_opt) + 0.18 * ((temp - t_opt) > 0) * abs(temp - t_opt)
    h_penalty = 1 + 0.04 * abs(hum - h_opt) + 0.10 * ((hum - h_opt) > 0) * abs(hum - h_opt)
    # Product of penalties to simulate compounding spoilage risk
    return max(1, int(base / (t_penalty * h_penalty) + np.random.normal(0, 0.75)))


rows = []
np.random.seed(42)
for prod in produce_types:
    for temp in np.random.randint(0, 28, 12):
        for hum in np.random.randint(60, 100, 6):
            shelf = shelf_life_decay(prod, temp, hum)
            rows.append({
                'Type': prod['Type'],
                'Temperature_C': temp,
                'Humidity_%': hum,
                'Shelf_Life_Days': shelf
            })

df = pd.DataFrame(rows)
df.to_csv("fruit_veg_shelf_life.csv", index=False)
print(df.head(20))
