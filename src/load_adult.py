
#pip install ucimlrepo

from ucimlrepo import fetch_ucirepo
import pandas as pd

adult = fetch_ucirepo(id=2)  # UCI ID for Adult Income
df = pd.concat([adult.data.features, adult.data.targets], axis=1)
df.to_csv("data/raw/adult.csv", index=False)

