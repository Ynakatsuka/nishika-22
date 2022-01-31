import pandas as pd


def preprocess1():
    train = pd.read_csv("data/input/train.csv")
    cite = pd.read_csv("data/input/cite_v2.csv")

    cite["path"] = cite["path"].apply(lambda x: f"cite_images/{x}")
    train["path"] = train["path"].apply(lambda x: f"apply_images/{x}")

    cite = cite[cite.gid.isin(train.cite_gid.unique())]
    cite["target"], _ = pd.factorize(cite["gid"])
    train = train.merge(
        cite, left_on="cite_gid", right_on="gid", how="left", suffixes=("", "_")
    )
    train.drop(["gid_", "path_", "cite_gid", "cite_path"], axis=1, inplace=True)
    train = train.append(cite).reset_index(drop=True)

    train.to_csv("data/output/simple_train_and_cite.csv", index=False)


def preprocess2():
    train = pd.read_csv("data/input/train.csv")
    cite = pd.read_csv("data/input/cite_v2.csv")

    cite["path"] = cite["path"].apply(lambda x: f"cite_images/{x}")
    train["path"] = train["path"].apply(lambda x: f"apply_images/{x}")

    cite["target"], _ = pd.factorize(cite["gid"])
    train = train.merge(
        cite, left_on="cite_gid", right_on="gid", how="left", suffixes=("", "_")
    )
    train.drop(["gid_", "path_", "cite_gid", "cite_path"], axis=1, inplace=True)
    train = train.append(cite).reset_index(drop=True)

    train.to_csv("data/output/train_and_cite.csv", index=False)


def preprocess3():
    test = pd.read_csv("data/input/test.csv")
    cite = pd.read_csv("data/input/cite_v2.csv")

    cite["path"] = cite["path"].apply(lambda x: f"cite_images/{x}")
    test["path"] = test["path"].apply(lambda x: f"apply_images/{x}")

    test_cite = test.append(cite).reset_index(drop=True)
    test_cite.to_csv("data/output/test_and_cite.csv", index=False)


def main():
    preprocess1()
    preprocess2()
    preprocess3()


if __name__ == "__main__":
    main()
