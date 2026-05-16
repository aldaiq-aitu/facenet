from models.facenet import FaceNet
from models.efficientnet_lite import EfficientNetLite0Face


def download_all_weights():
    failures = []

    print("Downloading FaceNet pretrained weights (VGGFace2)...")
    try:
        FaceNet(pretrained="vggface2")
        print("FaceNet weights downloaded successfully!")
    except Exception as exc:
        failures.append(f"FaceNet: {exc}")
        print(f"Error downloading FaceNet: {exc}")

    print("\nDownloading EfficientNet-Lite0 ImageNet backbone weights...")
    try:
        EfficientNetLite0Face(pretrained=True)
        print("EfficientNet-Lite0 backbone weights downloaded successfully!")
    except Exception as exc:
        failures.append(f"EfficientNet-Lite0: {exc}")
        print(f"Error downloading EfficientNet-Lite0: {exc}")

    if failures:
        raise RuntimeError("One or more weight downloads failed: " + "; ".join(failures))

    print("\nAll downloads finished successfully.")


if __name__ == "__main__":
    download_all_weights()
