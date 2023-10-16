import argparse
from pathlib import Path
import cv2
import mimetypes
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def extract_frames(video_path: Path, dest_path: Path, interval: int):
    # Open the video
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"failed to open {video_path}")
        return f"Error: Couldn't open video {video_path}."

    # Create destination directory for frames, named after the video
    frame_dir = dest_path / video_path.stem
    frame_dir.mkdir(parents=True, exist_ok=True)

    frame_num = 0
    cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cnt += 1
        if cnt % interval != 0:
            continue

        # Construct frame filename and save the frame
        frame_filename = frame_dir / f"{cnt:04d}.png"
        cv2.imwrite(str(frame_filename), frame)

        frame_num += 1

    cap.release()
    return f"Extracted {frame_num} frames from {video_path}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract frames from videos in source directory and save to destination directory."
    )
    parser.add_argument(
        "source", type=Path, help="Path to the source directory with videos"
    )
    parser.add_argument(
        "dest", type=Path, help="Path to the destination directory to save frames"
    )
    parser.add_argument("--interval", type=int, default=1)

    args = parser.parse_args()

    video_files = [vf for vf in args.source.iterdir() if mimetypes.guess_type(vf)[0] and mimetypes.guess_type(vf)[0].startswith('video')]

    # We'll define a wrapper function for updating tqdm
    def callback(msg):
        pbar.set_description(f"{msg}")
        pbar.update(1)

    with Pool(processes=8) as pool:
        pbar = tqdm(total=len(video_files), unit="video")
        for video_file in video_files:
            pool.apply_async(extract_frames, args=(video_file, args.dest, args.interval), callback=callback)
        pool.close()
        pool.join()
        pbar.close()
