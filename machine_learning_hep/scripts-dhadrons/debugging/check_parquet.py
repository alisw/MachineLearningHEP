import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""

file: check_parquet.py
brief: Examples of different checks on any parquet file produced by the MLHEP preprocessing steps.
usage: python check_parquet.py AnalysisResultsReco_fPt1_2.parquet
author: Maja Karwowska <mkarwowska@cern.ch>, Warsaw University of Technology
"""

def plot_parquet(df):
    print(df["fY"])
    print(df["fY"][~np.isinf(df["fY"])])

    ds_fin = df["fY"][~np.isinf(df["fY"])]

    fig = plt.figure(figsize=(20, 15))
    ax = plt.subplot(1, 1, 1)
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
    print(df.size)

    print(f"df mean\n{df.mean()}")

    print(f"df[0]\n{df.iloc[0]}")

    plot_parquet(df)

    df_sel = df[df["y_test_probxgboostbkg"] > 1.0]
    print(f"sel df bkg:\n{df_sel}")
    df_sel = df[df["y_test_probxgboostnon_prompt"] < 0.00]
    print(f"sel df non-prompt:\n{df_sel}")
    df_sel = df[df["y_test_probxgboostprompt"] < 0.00]
    print(f"sel df prompt:\n{df_sel}")

    # Valid only for data with saved results of ML application on Hyperloop
    #print(f'ML columns:\n{df["fMlBkgScore"]}\n{df["fMlPromptScore"]}\n{df["fMlNonPromptScore"]}')
    #df_sel = df[df["fMlBkgScore"] > 1.0]
    #print(f'df sel ML bkg:\n{df_sel["fMlBkgScore"]}')
    #df_sel = df[df["fMlNonPromptScore"] < 0.0]
    #print(f'df sel ML non-prompt:\n{df_sel["fMlNonPromptScore"]}')


if __name__ == '__main__':
    main()
