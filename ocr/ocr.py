from pytesseract import Output
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
def ocr_core(img):
    text = pytesseract.image_to_string(img)
    return text


img = cv2.imread("image.png")


def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# noise removal
def remove_noise(image):
    return cv2.medianBlur(image, 5)


# thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


img = get_grayscale(img)
img = thresholding(img)
img = remove_noise(img)
print(ocr_core(img))
# import cv2
# import pytesseract


# def ocr_core(img):
#     text = pytesseract.image_to_string(img)
#     return text


# img = cv2.imread("image.png")


# def get_grayscale(image):
#     return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# # noise removal
# def remove_noise(image):
#     return cv2.medianBlur(image, 5)


# # thresholding
# def thresholding(image):
#     return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


# img = get_grayscale(img)
# img = thresholding(img)
# img = remove_noise(img)

# results = {
#     "par_num": [0, 0, 1, 1, 1, 1, 1],
#     "line_num": [0, 0, 0, 1, 1, 1, 1],
#     "word_num": [0, 0, 0, 0, 1, 2, 3],
#     "left": [0, 26, 26, 26, 26, 110, 216],
#     "top": [0, 63, 63, 63, 63, 63, 63],
#     "width": [300, 249, 249, 249, 77, 100, 59],
#     "height": [150, 25, 25, 25, 25, 19, 19],
#     "conf": ["-1", "-1", "-1", "-1", 97, 96, 96],
#     "text": ["", "", "", "", "", "", ""],
# }

# for i in range(0, len(results["text"])):
#     x = results["left"][i]
#     y = results["top"][i]
#     w = results["width"][i]
#     h = results["height"][i]
#     text = results["text"][i]
#     conf = int(results["conf"][i])

#     if conf > 70:
#         text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
#         cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         cv2.putText(
#             img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2
#         )

# cv2.imshow("Image with OCR", img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
