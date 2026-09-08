from imagedl.modules.sources import BingImageClient


def main():
    client = BingImageClient(
        work_dir="/app/image_based_cdd/downloads",
    )

    image_infos = client.search(
        "dog",
        search_limits=10,
        num_threadings=1,
    )

    print(f"found: {len(image_infos)}")

    client.download(
        image_infos,
        num_threadings=1,
    )


if __name__ == "__main__":
    main()
