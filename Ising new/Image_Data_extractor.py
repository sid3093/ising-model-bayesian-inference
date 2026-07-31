import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# 1. Load the image
img_path = r"/Users/sid/Downloads/Ising new/panel_a_CrI3 (1).png"
panel_a = Image.open(img_path).convert("RGB")
img_arr = np.array(panel_a)

# 2. Calibration: Defining the main plot boundaries (excluding labels and insets)
# You might need to tweak these slightly based on your exact file crop
p_top, p_bottom = 20, 290  # Pixels from top/bottom
p_left, p_right = 60, 340  # Pixels from left/right

# Physical axis limits from the image
x_min, x_max = 0, 120   # T (K)
y_min, y_max = 0, 3     # M (10^-2 emu)

# 3. Target Colors (R, G, B)
COLOR_ZFC = (220, 30, 30)   # Red circles
COLOR_FC  = (50, 80, 160)   # Blue squares
TOLERANCE = 50

def get_data_from_mask(img, target_color):
    # Create mask only within the main plot box
    roi = img[p_top:p_bottom, p_left:p_right]
    mask = np.all(np.abs(roi.astype(int) - target_color) < TOLERANCE, axis=-1)
    
    # Get pixel coordinates
    ys, xs = np.where(mask)
    
    # Map to physical values
    # x physical = x_min + (pixel_x / roi_width) * range
    T = x_min + (xs / (p_right - p_left)) * (x_max - x_min)
    # y physical = y_max - (pixel_y / roi_height) * range (because y=0 is top)
    M = y_max - (ys / (p_bottom - p_top)) * (y_max - y_min)
    
    return T, M

# 4. Extract and Bin
T_zfc_raw, M_zfc_raw = get_data_from_mask(img_arr, COLOR_ZFC)
T_fc_raw,  M_fc_raw  = get_data_from_mask(img_arr, COLOR_FC)

def bin_data(T, M, bins=80):
    df = pd.DataFrame({'T': T, 'M': M})
    df['T_bin'] = pd.cut(df['T'], bins=np.linspace(x_min, x_max, bins))
    binned = df.groupby('T_bin', observed=True).median().reset_index()
    return binned['T'].values, binned['M'].values

T_zfc, M_zfc = bin_data(T_zfc_raw, M_zfc_raw)
T_fc,  M_fc  = bin_data(T_fc_raw,  M_fc_raw)

# 5. Save to CSV
# We use pandas for a clean, easy-to-load CSV format
pd.DataFrame({'Temp_K': T_zfc, 'Mag_emu_e2': M_zfc}).to_csv("CrI3_ZFC_Bulk.csv", index=False)
pd.DataFrame({'Temp_K': T_fc, 'Mag_emu_e2': M_fc}).to_csv("CrI3_FC_Bulk.csv", index=False)

# 6. Verification Plot
plt.figure(figsize=(8, 5))
plt.scatter(T_zfc, M_zfc, color='red', label='Extracted ZFC', s=10)
plt.scatter(T_fc, M_fc, color='blue', label='Extracted FC', s=10)
plt.axvline(61, color='black', linestyle='--', label='$T_c \approx 61$ K')
plt.title("Extracted $M$ vs $T$ for Bulk $CrI_3$")
plt.xlabel("Temperature (K)")
plt.ylabel("M ($10^{-2}$ emu)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Files saved: CrI3_ZFC_Bulk.csv, CrI3_FC_Bulk.csv")