"""
Initial Specifications
https://chatgpt.com/c/67f35474-da74-800a-8fec-73f5dc112683

Revised Functions
https://chatgpt.com/g/g-p-6767cb08c74881918472057248dcd16a-video/c/67f4ef93-09c8-800a-8c00-379433f260be

Face Continuity
https://chatgpt.com/g/g-p-6767cb08c74881918472057248dcd16a-video/c/67f725f1-2708-800a-9180-9bd29ec9aaf4
"""

import os
import cv2
import shutil
from gradio_client import Client
from moviepy.editor import VideoFileClip, concatenate_videoclips

# Import the Lab color-matching function
from color_utils import match_histograms_lut_lab

client = Client("http://localhost:7870")

def extract_last_frame(video_path, output_image_path):
    """
    Extract the final frame from a video file.
    """    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    success, frame = cap.read()
    cap.release()

    if not success:
        raise RuntimeError(f"Failed to extract the last frame from {video_path}.")
    
    cv2.imwrite(output_image_path, frame)
    return output_image_path

def enforce_resolution(frame_path, target_w=720, target_h=1280):
    frame = cv2.imread(frame_path)
    resized = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(frame_path, resized)

def check_if_frame_has_face(
    image_path, face_cascade_path_rel="haarcascade_frontalface_default.xml"
):
    """
    Returns True if the given `image_path` contains at least one face, False otherwise.
    """
    face_cascade_path_abs = cv2.data.haarcascades + face_cascade_path_rel
    face_cascade = cv2.CascadeClassifier(face_cascade_path_abs)
    
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0

def chain_snippets(input_dir, output_file="final_video.mp4"):
    """
    Concatenate all MP4 files in input_dir into a single video file, in sorted order.
    """
    # Grab all .mp4 files in the directory
    files = [
        f for f in os.listdir(input_dir) 
        if f.endswith(".mp4") 
        and not f.startswith("._") # Exclude resource forks
    ]
    # Sort them by name or by snippet index
    files.sort()
    
    # Build the full paths
    snippet_paths = [os.path.join(input_dir, f) for f in files]
    print("Will concatenate these snippet files:\n", snippet_paths)

    if not snippet_paths:
        raise RuntimeError("No mp4 clips found to concatenate.")

    # Load each into MoviePy
    clips = [VideoFileClip(p) for p in snippet_paths]
    
    # Concatenate
    final_clip = concatenate_videoclips(clips)

    # Write final result
    final_clip.write_videofile(output_file, codec="libx264", fps=30)
    print(f"Created {output_file}")

# ---------------------------------
# Example generation + chaining
# ---------------------------------
def generate_chained_video_snippets(
    max_tries=5,
    num_snippets=5,
    snippet_dir="long_video_snippets",
    prompt=(
        "A teenage woman who is around 18 years old and is Singapore Chinese. "
        "She is wearing a red, velvety jacket. She has headphones resting around her neck. "
        "She stands still in a softly lit living room with light streaming in through a wide window. "
        "She faces to the front. She speaks to the front. The skyline of a city is faintly visible through the glass."
        "She consistently faces front, making sure her face is always visible."
        "The lighting remains identical to the previous frames, with consistent exposure and color temperature."
        "No drastic changes in brightness or hue across frames."
        "Soft, warm indoor lighting with no color shifts."
    ),
    negative_prompt="She turns away from the camera. She turns around. She faces away. She looks to the back.",
    resolution="720p",
    video_length=50,          # For each snippet
    seed=-1,
    num_inference_steps=50,
    guidance_scale=1.0,
    flow_shift=7,
    embedded_guidance_scale=6.0,
    repeat_generation=1,
    tea_cache=0.1,
    loras_choices=[],
    loras_mult_choices="",
    max_frames=9,
    riflex_setting=0,
    stability_setting=0,
    initial_image_path="/home/chee-wei-jie/Desktop/HunyuanVideoGP/i2v_images/headphones_front.jpg"
):
    """
    Generates multiple short T2V snippets in a loop, each continuing from the final frame of the previous one,
    then saves them to snippet_dir, and finally merges them into a single mp4 file.
    """
    # 1) Ensure snippet_dir exists
    os.makedirs(snippet_dir, exist_ok=True)

    # 2) We start from an image for the first snippet:
    image_to_continue = {"path": initial_image_path} if initial_image_path else None
    video_to_continue = None  # Starting with a static image

    # We'll keep track of the "reference frame" from the previous snippet
    prev_snippet_frame_path = None

    # We'll store the final snippet paths in a list (helpful if you want to do
    # something custom at the end).
    snippet_paths = []    

    for idx in range(num_snippets):
        print(f"=== Generating snippet #{idx+1} ===")
        
        for attempt in range(max_tries):
            # 1) Call the generate_video endpoint
            result_status = client.predict(
                prompt,
                negative_prompt,
                resolution,
                video_length,
                seed,
                num_inference_steps,
                guidance_scale,
                flow_shift,
                embedded_guidance_scale,
                repeat_generation,
                tea_cache,
                loras_choices,
                loras_mult_choices,
                image_to_continue,
                video_to_continue,
                max_frames,
                riflex_setting,
                stability_setting,
                api_name="/generate_video",
            )
            print("Status from generate_video:", result_status)

            # -- Step B: Immediately call /refresh_gallery to get the snippet path
            gallery_items = client.predict(api_name="/refresh_gallery")
            if not gallery_items:
                raise RuntimeError("No items returned by /refresh_gallery.")
            
            print("GALLERY_ITEMS")
            print(gallery_items)
            print()

            # The newest item is often at the end of the list:
            last_item = gallery_items[-1]
            if "video" not in last_item:
                raise RuntimeError("The last gallery item is not a video.")

            generated_video_path = last_item["video"]
            print(f"Snippet #{idx+1} saved at: {generated_video_path}")

            print("GENERATED_VIDEO_PATH")
            print(generated_video_path)
            print()

            # -- Step C: Save (move/copy) the snippet to snippet_dir with a meaningful name
            # e.g. "snippet_001.mp4", "snippet_002.mp4", ...
            snippet_filename = f"snippet_{idx+1:03d}.mp4"  # 3-digit padded index
            snippet_full_path = os.path.join(snippet_dir, snippet_filename)

            print("SNIPPET_FULL_PATH")
            print(snippet_full_path)
            print()

            # Now move or copy from the auto-generated path to our desired location.
            # (You can do shutil.move(...) if you don't need the original file.)
            shutil.copy2(generated_video_path, snippet_full_path)
            print(f"Snippet saved as {snippet_full_path}")
            snippet_paths.append(snippet_full_path)

            # -- Step D: Extract the last frame => feed it into the next snippet
            last_frame_file = f"last_frame_{idx+1}.jpg"
            last_frame_path = os.path.join(snippet_dir, last_frame_file)
            extract_last_frame(snippet_full_path, last_frame_path)

            # -- Step E: Optional: Enforce 720x1280 Resolution
            enforce_resolution(last_frame_path, 720, 1280)

            # -- Step F: Optional: Color-match the newly generated last frame
            # to the previous snippet's last frame (if exists).
            if prev_snippet_frame_path is not None:
                source_img = cv2.imread(last_frame_path)
                reference_img = cv2.imread(prev_snippet_frame_path)

                # Call the imported LUT matching function
                matched_img = match_histograms_lut_lab(
                    source=source_img, 
                    reference=reference_img,
                    match_l=True,   # match the L channel
                    match_ab=True   # match the a/b channels
                )

                cv2.imwrite(last_frame_path, matched_img)
                print(f"Color-matched the last frame of snippet #{idx+1} to snippet #{idx}'s last frame.")

            # -- Step G: Check if face is present
            has_face = check_if_frame_has_face(last_frame_path)
            if has_face:
                # Good snippet => break out of the re-gen loop
                print("Found a face in the last frame. Proceeding.")
                break
            else:
                print(f"No face found in attempt #{attempt+1}. Re-generating...")

                # Optionally delete snippet to avoid clutter
                os.remove(snippet_full_path)
                os.remove(last_frame_path)

        # If you exit the for-attempt loop without a face, handle it:
        if not has_face:
            print("Max attempts reached. Using snippet without a face or skipping.")
            # Decide how you want to handle a repeated failure
            raise RuntimeError("Could not generate snippet with face in the last frame.")

        # F) Update references for next iteration
        prev_snippet_frame_path = last_frame_path
        image_to_continue = {"path": last_frame_path}
        video_to_continue = None  # or pass the entire video if desired

    # 3) Now that we have multiple short snippets in snippet_dir, chain them
    output_file = "final_chained_video.mp4"
    chain_snippets(snippet_dir, output_file)
    print(f"Done! Final video is {output_file}")

# -----------------------------------------------------------------------------
# Usage Example
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    generate_chained_video_snippets(
        num_snippets=1, 
        video_length=24,
        num_inference_steps=10,
    )