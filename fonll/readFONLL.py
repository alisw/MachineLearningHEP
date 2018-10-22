import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv('fo_pp_d0meson_5TeV_y0p5.csv')
df_pt= df["pt"]
print (df_pt)
plt.hist(df_pt)
plt.show()
