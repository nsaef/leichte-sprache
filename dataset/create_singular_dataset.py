import os
import uuid

import pandas as pd


def load_konvens(dirname: str) -> pd.DataFrame:
    """
    Load the konvens dataset file(s) and parse them into a standardized format. Combine sentences that belong to the same text
    and preserve the original IDs. Returns a dataframe containing a UUID, the combined text and the original IDs.
    :param dirname: path of the directory containing the konvens file(s) relative to the root directory
    """
    filenames = os.listdir(dirname)
    data = []
    for fname in filenames:
        if not fname.startswith("konvens_"):
            continue
        fpath = f"{dirname}/{fname}"
        df = pd.read_csv(fpath)
        topics = set(df.topic)
        for topic in topics:
            text = "\n".join(list(df[df.topic == topic].phrase)).replace(
                " \\newline ", "\n"
            )
            orig_ids = ",".join(list(df[df.topic == topic]["sent-id"].astype(str)))
            data.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": text,
                    "orig_ids": orig_ids,
                }
            )
    data_df = pd.DataFrame(data)
    return data_df


def main():
    # todo: docstring
    # todo: more flexible path handling
    dirname = "dataset/files"
    datasets = []

    konvens_data = load_konvens(dirname)
    datasets.append(konvens_data)

    # todo: combine all available datasets into one, store as HF dataset
    df = pd.concat(datasets) if len(datasets) > 1 else datasets[0]
    df.to_csv(f"{dirname}/dataset_singular.csv", index=False)
    return


if __name__ == "__main__":
    main()
