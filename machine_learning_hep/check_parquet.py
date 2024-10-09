import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_parquet(df):
    print(df["fY"])
    print(df["fY"][~np.isinf(df["fY"])])

    ds_fin = df["fY"][~np.isinf(df["fY"])]

    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
    #ax.set_xlim([0, (df["fY"].mean()*2)])
    plt.hist(ds_fin.values, bins=50)
    ax.set_xlabel("fY", fontsize=30)
    ax.set_ylabel("Entries", fontsize=30)
    fig.savefig("fY.png", bbox_inches='tight')
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="file to process")
    args = parser.parse_args()

    df = pd.read_parquet(args.infile)
    print(f"df columns: {df.columns}")
    print(f"full df:\n{df}")

    print(f"df mean\n{df.mean()}")

    print(f"df[0]\n{df.iloc[0]}")

    df_sel = df[df["y_test_probxgboostbkg"] <= 0.02]
    print(f"sel df:\n{df_sel}")
    #df_sel = df_sel[df_sel["y_test_probxgboostnon_prompt"] <= 0.08]
    #print(f"sel df non-prompt:\n{df_sel}")

    #print(f'ML columns:\n{df["fMlBkgScore"]}\n{df["fMlPromptScore"]}\n{df["fMlNonPromptScore"]}')
    #df_sel = df[df["fMlBkgScore"] >= 0.02]
    #df_sel = df[df["fMlNonPromptScore"] < 0.15]
    #print(f'df sel ML columns:\n{df_sel["fMlBkgScore"]}\n{df_sel["fMlNonPromptScore"]}')


if __name__ == '__main__':
    main()
