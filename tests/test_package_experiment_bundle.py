from pathlib import Path

from scripts.package_experiment_bundle import main as package_main


def test_package_experiment_bundle(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "splits" / "train" / "person_0001").mkdir(parents=True)
    (tmp_path / "data" / "pairs").mkdir(parents=True)
    (tmp_path / "data" / "pairs" / "val_pairs.csv").write_text("path1,path2,is_same\n", encoding="utf-8")
    (tmp_path / "data" / "pairs" / "test_pairs.csv").write_text("path1,path2,is_same\n", encoding="utf-8")

    monkeypatch.setattr("sys.argv", ["package_experiment_bundle.py"])
    package_main()

    assert (tmp_path / "shared_experiment_bundle" / "bundle_manifest.json").exists()
    assert (tmp_path / "shared_experiment_bundle" / "data" / "pairs" / "val_pairs.csv").exists()
