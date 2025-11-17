# ~/Desktop/hunyuanvideo_client.py

from gradio_client import Client

client = Client("http://localhost:7870/")

# For T2V usage with defaults in your code:
prompt = "A teenage woman who is around 18 years old and is Singapore Chinese. She is wearing a red, velvety jacket. She has headphones resting around her neck. She stands still in a softly lit living room with light streaming in through a wide window.  She faces to the front. She speaks to the front. The skyline of a city is faintly visible through the glass. There is a subtle depth in the scene, with a hint of furniture and decoration in the background."
negative_prompt = ""
resolution = "720p"  # must be one of ['720p', '540p', '360p']
video_length = 130
seed = -1
num_inference_steps = 30
guidance_scale = 1.0
flow_shift = 7
embedded_guidance_scale = 6.0
repeat_generation = 30
tea_cache = 0.1              # must be from [0, 0.1, 0.15]
loras_choices = []           # no LoRAs
loras_mult_choices = ""      # no multipliers

image_to_continue = {
  "path": "/home/chee-wei-jie/Desktop/HunyuanVideoGP/i2v_images/headphones_front.jpg"
}     # for T2V mode, we can pass None
video_to_continue = None     # also None

# image_to_continue = None
# video_to_continue = {
#   "video": "/home/chee-wei-jie/Desktop/HunyuanVideoGP/i2v_images/headphones_front.mp4"
# }

max_frames = 9
riflex_setting = 0         # 0, 1, or 2
stability_setting = 0      # 0, 1, or 2

result = client.predict(
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
    api_name="/generate_video"
)
print("Response:", result)
