import argparse
import zipfile

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("elife_file")
    parser.add_argument("plos_file")

    args = parser.parse_args()

    elife_df = pd.read_feather(args.elife_file)["summary"]
    plos_df = pd.read_feather(args.plos_file)["summary"]

    elife_txt = "elife.txt"
    plos_txt = "plos.txt"

    elife_df.to_csv(elife_txt, sep="\t", index=False, header=False)
    plos_df.to_csv(plos_txt, sep="\t", index=False, header=False)

    zip_filename = "output.zip"
    print(f"Creating {zip_filename}...")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(elife_txt)
        zipf.write(plos_txt)

    print(f"Successfully created {zip_filename} containing {elife_txt} and {plos_txt}")


if __name__ == "__main__":
    main()
