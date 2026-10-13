from huggingface_hub import snapshot_download

DATASET = "data/pusht_keypoints/data/chunk-000/file-000.parquet"

def main():
    print("Hello from robot-learning-from-scratch!")

    # path = snapshot_download(
    #     repo_id="lerobot/pusht_keypoints",
    #     repo_type="dataset",
    #     local_dir="data/pusht_keypoints",
    # )
    # print(path)


if __name__ == "__main__":
    main()
