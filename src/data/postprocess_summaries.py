import argparse
import re

from .prepare_submission import read_files, create_submission


def remove_brackets(text):
    text = re.sub(r"\([^()]*\)|\{[^{}]*\}|\[[^\[\]]*\]", "", text)
    return text


def remove_cutoff_sentences(text):
    if text.endswith("."):
        return text

    last_period_index = text.rfind(".")

    if last_period_index != -1:
        return text[: last_period_index + 1]

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("elife_file")
    parser.add_argument("plos_file")

    args = parser.parse_args()
    elife_df, plos_df = read_files(args)

    # Remove cutoff sentences
    elife_df = elife_df.apply(remove_cutoff_sentences)
    plos_df = plos_df.apply(remove_cutoff_sentences)

    # Remove brackets ()|{}|[]
    elife_df = elife_df.apply(remove_brackets)
    plos_df = plos_df.apply(remove_brackets)

    create_submission(elife_df, plos_df)


if __name__ == "__main__":
    main()
