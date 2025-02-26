import textgrids
import pandas as pd
import argparse
import os


def extract_intervals(
    textgrid_path,
    tier_a_name,
    tier_b_name,
    whitelist_phones=[],
    blacklist_words=[],
    minimum_duration=0,
):
    tg = textgrids.TextGrid(textgrid_path)

    tier_a = tg[tier_a_name]
    tier_b = tg[tier_b_name]

    data = []
    for interval_a in tier_a:
        start_a, end_a, label_a = (
            interval_a.xmin,
            interval_a.xmax,
            interval_a.text.strip(),
        )
        if label_a not in whitelist_phones:
            continue
        if (end_a - start_a) < minimum_duration:
            continue

        label_b = None
        for interval_b in tier_b:
            if interval_b.xmin <= start_a and interval_b.xmax >= end_a:
                label_b = interval_b.text.strip()
                break
        if label_b and label_b in blacklist_words:
            continue

        basename = os.path.splitext(os.path.basename(textgrid_path))[0]
        data.append([basename, start_a, end_a, label_a, label_b])

    return pd.DataFrame(data, columns=["filename", "start", "end", "phone_label", "word"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract intervals from a Praat TextGrid."
    )
    parser.add_argument("tier_a_name", help="Name of phone tier.")
    parser.add_argument("tier_b_name", help="Name of word tier.")
    parser.add_argument("output_csv", help="Path to save the output CSV.")
    parser.add_argument("textgrid_paths", help="Paths to the TextGrid files", nargs="+")
    parser.register("type", "comma-separated list", lambda s: s.split(","))
    parser.add_argument(
        "--whitelist-phones",
        help="Comma-separated list of phones to include",
        default=[],
        type="comma-separated list",
    )
    parser.add_argument(
        "--blacklist-words",
        help="Comma-separated list of words to exclude",
        default=[],
        type="comma-separated list",
    )
    parser.add_argument(
        "--minimum-duration", help="Minimum duration in seconds", default=0, type=float
    )

    args = parser.parse_args()

    intervals = pd.concat(
        [
            extract_intervals(
                fn,
                args.tier_a_name,
                args.tier_b_name,
                args.whitelist_phones,
                args.blacklist_words,
                args.minimum_duration,
            )
            for fn in args.textgrid_paths
        ]
    )
    intervals.to_csv(args.output_csv, index=False)
