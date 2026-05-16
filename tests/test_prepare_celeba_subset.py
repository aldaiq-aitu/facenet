from pathlib import Path

from PIL import Image

from scripts.prepare_celeba_subset import main as prepare_main


def test_prepare_celeba_subset(tmp_path, monkeypatch):
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    identity_file = tmp_path / "identity.txt"
    lines = []
    for identity in range(1, 4):
        for image_idx in range(1, 4):
            name = f"{identity:02d}_{image_idx:02d}.jpg"
            Image.new("RGB", (10, 10)).save(images_dir / name)
            lines.append(f"{name} {identity}")
    identity_file.write_text("\n".join(lines), encoding="utf-8")
    output_dir = tmp_path / "subset"

    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_celeba_subset.py",
            "--images-dir",
            str(images_dir),
            "--identity-file",
            str(identity_file),
            "--output-dir",
            str(output_dir),
            "--num-identities",
            "2",
            "--images-per-identity",
            "2",
            "--min-images-per-identity",
            "3",
        ],
    )
    prepare_main()

    created = list(output_dir.glob("person_*/*.jpg"))
    assert len(created) == 4
    assert (output_dir / "celeba_subset_manifest.csv").exists()
