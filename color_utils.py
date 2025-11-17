import cv2
import numpy as np

def match_histograms_lut_lab(
    source, 
    reference, 
    match_l=True, 
    match_ab=True
):
    """
    Perform LUT-based histogram matching in Lab color space.
    - source: BGR (8-bit) image to adjust
    - reference: BGR (8-bit) image whose histogram we want to match
    - match_l: Whether to match the L (lightness) channel
    - match_ab: Whether to match the a and b (color) channels
    
    Returns a new BGR image with matched color distribution.
    """

    # 1) Convert to Lab
    src_lab = cv2.cvtColor(source, cv2.COLOR_BGR2Lab)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_BGR2Lab)

    src_l, src_a, src_b = cv2.split(src_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)

    def compute_cdf(hist):
        cdf = hist.cumsum()
        return cdf / cdf[-1]  # normalized to [0,1]

    def create_lut(src_cdf, ref_cdf):
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(256):
            s_val = src_cdf[i]
            # Locate the intensity in the reference distribution
            # with a cumulative probability closest to s_val
            j = np.searchsorted(ref_cdf, s_val, side='left')
            j = np.clip(j, 0, 255)
            lut[i] = j
        return lut

    def match_channel(src_chan, ref_chan):
        # Compute histograms
        hist_src = cv2.calcHist([src_chan], [0], None, [256], [0, 256]).flatten()
        hist_ref = cv2.calcHist([ref_chan], [0], None, [256], [0, 256]).flatten()

        src_cdf = compute_cdf(hist_src)
        ref_cdf = compute_cdf(hist_ref)

        lut = create_lut(src_cdf, ref_cdf)
        return cv2.LUT(src_chan, lut)

    # 2) Match each channel if needed
    if match_l:
        matched_l = match_channel(src_l, ref_l)
    else:
        matched_l = src_l

    if match_ab:
        matched_a = match_channel(src_a, ref_a)
        matched_b = match_channel(src_b, ref_b)
    else:
        matched_a = src_a
        matched_b = src_b

    # 3) Merge channels back and convert to BGR
    matched_lab = cv2.merge([matched_l, matched_a, matched_b])
    matched_bgr = cv2.cvtColor(matched_lab, cv2.COLOR_Lab2BGR)

    return matched_bgr
