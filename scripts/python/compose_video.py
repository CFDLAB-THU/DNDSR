import os
import re
import sys
import cv2
import argparse


def get_png_files(base_dir, regex_pattern):
    """Get sorted list of .png files matching the regex pattern."""
    pattern = re.compile(regex_pattern)
    png_files = [
        f for f in os.listdir(base_dir) if f.endswith(".png") and pattern.match(f)
    ]
    png_files.sort()
    return png_files


def resize_image(image, target_size):
    """Resize image to the specified target size."""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def create_video_from_frames(frames, output_path, fps=30):
    """Create a video from a sequence of frames."""
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    out.release()


def main():
    parser = argparse.ArgumentParser(description="Create a video from .png images.")
    parser.add_argument("base_dir", help="Base directory containing .png files")
    parser.add_argument("regex_pattern", help="Regex pattern to match .png file names")
    parser.add_argument("output_path", help="Output video file path", default=None)
    parser.add_argument("--fps", help="output fps", type=float, default=10)
    args = parser.parse_args()

    output_path = args.output_path
    if output_path is None:
        output_path = os.path.join(args.base_dir, "out.mp4")

    # Step 1: Get the sorted list of .png files matching the regex
    png_files = get_png_files(args.base_dir, args.regex_pattern)
    print(f"Found Files:\n" + "".join([f"{fname}\n" for fname in png_files]))
    if not png_files:
        print("No .png files matching the pattern were found.")
        sys.exit(1)

    # Step 2: Read and process the images
    frames = []
    first_image = cv2.imread(os.path.join(args.base_dir, png_files[0]))
    if first_image is None:
        print(f"Error reading the first image: {png_files[0]}")
        sys.exit(1)
    target_size = (first_image.shape[1], first_image.shape[0])  # (width, height)
    print(f"=== Target size: {target_size}")

    for file_name in png_files:
        image_path = os.path.join(args.base_dir, file_name)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error reading image: {file_name}")
            continue
        resized_image = resize_image(image, target_size)
        frames.append(resized_image)

    if not frames:
        print("No valid images were processed.")
        sys.exit(1)

    # Step 3: Create video from frames
    create_video_from_frames(frames, args.output_path, fps=args.fps)
    print(f"Video saved as \n{args.output_path}")


if __name__ == "__main__":
    main()
