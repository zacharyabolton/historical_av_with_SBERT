import re
import pandas as pd
from bs4 import BeautifulSoup

RE_P1P2 = re.compile(r"optimal p1/p2:\s*([\d.]+)\s+([\d.]+)")
RE_SCORES = {
    "auc": re.compile(r"'auc': np\.float64\(([\d.]+)\)"),
    "c@1": re.compile(r"'c@1': ([\d.]+)"),
    "f_05_u": re.compile(r"'f_05_u': ([\d.]+)"),
    "F1_opt": re.compile(r"'F1': np\.float64\(([\d.]+)\)"),
    "brier": re.compile(r"'brier': np\.float64\(([\d.]+)\)"),
    "overall": re.compile(r"'overall': np\.float64\(([\d.]+)\)")
}
RE_DEV = re.compile(r"Dev results.*F1=([\d.]+)\s*at th=([\d.\-e]+)")


def parse_training_metrics(html_file):
    with open(html_file, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")

    dataset_name = None
    results = []

    for tag in soup.find_all(["h2", "h3", "pre"]):
        if tag.name == "h2":
            dataset_name = tag.get_text(strip=True)

        elif tag.name == "h3" and tag.get_text(strip=True)\
                .startswith("k_"):
            k_val = tag.get_text(strip=True)

            p1, p2 = None, None
            auc = c_at_1 = f05u = F1_opt = brier = overall = None
            F1_dev = None
            th_dev = None

            continue

        elif tag.name == "pre":
            text = tag.get_text()
            if "optimal p1/p2:" in text:
                match_p1p2 = RE_P1P2.search(text)
                if match_p1p2:
                    p1 = float(match_p1p2.group(1))
                    p2 = float(match_p1p2.group(2))

                extracted_scores = {}
                for key, pattern in RE_SCORES.items():
                    mm = pattern.search(text)
                    if mm:
                        extracted_scores[key] = float(mm.group(1))
                    else:
                        extracted_scores[key] = None

                auc = extracted_scores["auc"]
                c_at_1 = extracted_scores["c@1"]
                f05u = extracted_scores["f_05_u"]
                F1_opt = extracted_scores["F1_opt"]
                brier = extracted_scores["brier"]
                overall = extracted_scores["overall"]

                match_dev = RE_DEV.search(text)
                if match_dev:
                    F1_dev = float(match_dev.group(1))
                    th_str = match_dev.group(2)
                    try:
                        th_dev = float(th_str)
                    except:
                        th_dev = None

                results.append({
                    "Dataset": dataset_name,
                    "k": k_val,
                    "p1": p1,
                    "p2": p2,
                    "auc": auc,
                    "c@1": c_at_1,
                    "f_05_u": f05u,
                    "F1_opt": F1_opt,
                    "brier": brier,
                    "overall": overall,
                    "F1_dev": F1_dev,
                    "th_dev": th_dev
                })

    df = pd.DataFrame(results)

    def extract_k_num(k_str):
        try:
            return int(k_str.replace("k_", ""))
        except:
            return None

    df["k_num"] = df["k"].apply(extract_k_num)
    df.sort_values(by=["Dataset", "k_num"], inplace=True)
    df.drop("k_num", axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


df_metrics = parse_training_metrics("training_metrics.html")
df_metrics.to_csv('training_metrics.csv', index=False)