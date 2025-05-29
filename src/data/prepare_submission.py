"""Script to prepare model outputs for submission."""

import argparse
import zipfile
from typing import Tuple

import pandas as pd


def read_files(cli_args: argparse.Namespace) -> Tuple[pd.Series, pd.Series]:
    """Read summary data from feather files."""
    elife_df = pd.read_feather(cli_args.elife_file)["summary"]
    plos_df = pd.read_feather(cli_args.plos_file)["summary"]

    return elife_df, plos_df


def create_submission(elife_df: pd.Series, plos_df: pd.Series) -> None:
    """Create submission package from summary data."""
    elife_txt = "elife.txt"
    plos_txt = "plos.txt"

    # Save summaries to text files
    elife_df.to_csv(elife_txt, sep="\t", index=False, header=False)
    plos_df.to_csv(plos_txt, sep="\t", index=False, header=False)

    # Create zip file
    zip_filename = "output.zip"
    print(f"Creating {zip_filename}...")

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(elife_txt)
        zipf.write(plos_txt)

    print(f"Successfully created {zip_filename} containing {elife_txt} and {plos_txt}")


def main():
    """Main function to process files and create submission package."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "elife_file"
    )
    parser.add_argument(
        "plos_file"
    )

    args = parser.parse_args()
    elife_df, plos_df = read_files(args)
    create_submission(elife_df, plos_df)


if __name__ == "__main__":
    main()
