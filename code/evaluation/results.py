import os
from os.path import join


def clean_dataset_names(dataset_name):
    return dataset_name.replace(" ", "_").lower()


def remove_newline_at_eof(file):
    with open(file, "r") as f:
        contents = f.read()
    with open(file, "w") as f:
        f.write(contents.rstrip(os.linesep))


def generate_model_comparison_by_dataset(results_df, output_dir):
    models_metrics_best = results_df[results_df["contamination"] == 0][
        ["dataset", "model", "fbeta", "precision", "recall"]
    ].pivot_table(index="model", columns="dataset", aggfunc="max")

    models_metrics_best.columns = models_metrics_best.columns.swaplevel(0, 1)
    models_metrics_best.sort_index(axis=1, level=0, inplace=True)
    models_metrics_best.columns = [
        "_".join(col).rstrip() for col in models_metrics_best.columns.values
    ]

    output_file = join(output_dir, "model_comparison_on_datasets.csv")
    models_metrics_best.to_csv(output_file, float_format="%.3g")
    remove_newline_at_eof(output_file)


def generate_dataset_contamination_model_comparison(results_df, output_dir):
    df = results_df.copy()
    df["contamination"] = (df["contamination"] * 100).astype(int)
    contaminations_metrics = df[
        [
            "dataset",
            "model",
            "contamination",
            "fbeta",
            "precision",
            "recall",
            "true_positives",
        ]
    ].pivot_table(index="contamination", columns=["dataset", "model"])

    contaminations_metrics.columns = contaminations_metrics.columns.swaplevel(
        0, 1
    ).swaplevel(1, 2)
    contaminations_metrics.sort_index(axis=1, level=0, inplace=True)
    contaminations_metrics.sort_index(axis=0, level=1, inplace=True)

    datasets = set(
        [dataset for (dataset, model, _) in list(iter(contaminations_metrics.columns))]
    )
    for dataset in datasets:
        output_file = join(output_dir, f"contmination_comparison_{dataset}.csv")
        cont_df = contaminations_metrics[dataset]
        cont_df.columns = ["_".join(col).rstrip() for col in cont_df.columns.values]
        cont_df = cont_df.sort_values(by=["contamination"], ascending=True)
        cont_df.to_csv(output_file, float_format="%.3g")
        remove_newline_at_eof(output_file)
