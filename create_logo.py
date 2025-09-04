import cv2
import numpy as np

# Create a simple logo placeholder
logo = np.ones((100, 100, 3), dtype=np.uint8) * 255

# Add some text
cv2.putText(logo, "LOGO", (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 50, 50), 2)

# Add border
cv2.rectangle(logo, (5, 5), (94, 94), (100, 100, 100), 2)

cv2.imwrite("assets/logo.png", logo)
print("Logo created at assets/logo.png")