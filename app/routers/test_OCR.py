import pytesseract
from bidi.algorithm import get_display
from PIL import Image

# Load your image
img = Image.open("test_arabic.png")

# Run OCR with Arabic language
raw_text = pytesseract.image_to_string(img, lang='ara', config='--oem 1 --psm 6')

print(raw_text)

