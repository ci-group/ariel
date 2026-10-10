import os
from moviepy import VideoFileClip, clips_array

def combine_videos_from_directory(directory, output_file="combined_output.mp4"):
    """
    Combines specific videos from a directory into a single video and saves it.

    Args:
        directory (str): Path to the directory containing video files.
        output_file (str): Name of the output video file.
    """
    # Define the filenames to look for
    filenames = {
        "vid1": "iso_view.mp4",
        "vid2": "top_view.mp4",
        "vid3": "speed_plot.mp4",
        "vid4": "angular_speed_plot.mp4",
        "long_vid": "actions_plot.mp4"
    }

    # Resolve full paths for the required videos
    video_paths = {key: os.path.join(directory, value) for key, value in filenames.items()}

    # Load the video files
    try:
        vid1 = VideoFileClip(video_paths["vid1"])
        vid2 = VideoFileClip(video_paths["vid2"])
        vid3 = VideoFileClip(video_paths["vid3"])
        vid4 = VideoFileClip(video_paths["vid4"])
        long_vid = VideoFileClip(video_paths["long_vid"])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Arrange square videos in 2x2 grid
    grid = clips_array([[vid1, vid3],
                        [vid2, vid4]])

    # Combine the grid and long video side by side
    final_video = clips_array([[grid, long_vid]])

    # Write the result to the output file
    final_video.write_videofile(directory+"/"+output_file, codec='libx264', audio_codec='aac')

# Example usage
if __name__ == "__main__":
    # Specify the directory containing the videos
    # directory = "/path/to/your/videos"
    # For demonstration, using a hardcoded path
    # You should replace this with your actual path
    combine_videos_from_directory("/home/jed/workspaces/airevolve/plots/example_comparison")