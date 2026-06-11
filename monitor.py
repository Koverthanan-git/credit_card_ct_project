import pandas as pd
import numpy as np
# FIX Bug 1: Evidently 0.7.x restructured its API.
# Old: from evidently.report import Report / from evidently.metric_preset import DataDriftPreset
# New: top-level evidently package + evidently.presets
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


def evaluate_production_drift():
    # 1. Load your original baseline training parameters
    reference_df = pd.read_csv("reference_data.csv")

    # 2. Simulate live production incoming traffic logs
    # To demonstrate true drift, we shift the mean values of the incoming data distributions
    np.random.seed(99)
    drifted_features = np.random.randn(200, 30) + 1.5  # Adding a shift factor of +1.5
    production_df = pd.DataFrame(drifted_features, columns=[f"V{i}" for i in range(1, 31)])

    # FIX Bug 2: Evidently 0.7.x Report.run() requires Dataset objects, not raw DataFrames.
    # Old (broken): Report(metrics=[...]).run(reference_data=df, current_data=df)
    # New: Report([...]).run(current_dataset, reference_dataset) -> returns Snapshot
    report = Report([DataDriftPreset()])
    ref_ds = Dataset.from_pandas(reference_df.drop(columns=["Class"]))
    cur_ds = Dataset.from_pandas(production_df)
    snapshot = report.run(cur_ds, ref_ds)

    # 4. Parse output payload to detect percentage of drifting features
    # Old (broken): report_dict["metrics"][0]["result"]["dataset_drift"]
    # New: snapshot.dict()["metrics"][0]["value"]["share"] gives fraction of drifted columns
    snap_dict = snapshot.dict()
    drift_share = snap_dict["metrics"][0]["value"]["share"]  # 0.0 – 1.0
    dataset_drift = drift_share >= 0.5  # True if more than 50% of columns are drifting

    # Save human-readable structural analysis dashboard locally
    snapshot.save_html("drift_dashboard.html")

    return dataset_drift, production_df


if __name__ == "__main__":
    drift_detected, _ = evaluate_production_drift()
    print(f"Is Dataset Drift Active? -> {drift_detected}")