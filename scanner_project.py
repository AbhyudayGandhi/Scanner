import cv2
import os
import pytesseract

pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

cropping = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropped_image = None

# Mouse cropping
def crop_image(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, cropping, cropped_image

    if event == cv2.EVENT_LBUTTONDOWN:
        cropping = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            x_end, y_end = x, y
            temp_image = param.copy()
            cv2.rectangle(temp_image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Crop", temp_image)

    elif event == cv2.EVENT_LBUTTONUP:
        cropping = False
        x_end, y_end = x, y
        if x_start != x_end and y_start != y_end:
            cropped_image = param[y_start:y_end, x_start:x_end]
            cv2.imshow("Cropped", cropped_image)

# OCR
def process_with_tesseract(image_path, lang='eng+rus+spa+chi_sim'):
    # Read saved image
    img = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # confidence
    h_img, w_img, _ = img_gray.shape
    img_letters = img.copy()
    char_boxes = pytesseract.image_to_boxes(img_gray, lang=lang)
    conf_data = pytesseract.image_to_data(img_gray, lang=lang)
    for line in conf_data.splitlines()[1:]:
        parts = line.split()
        if len(parts) == 12:  
            x, y, w, h, conf = int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9]), float(parts[10])
            text = parts[11]
            if conf > 0:  
                cv2.putText(img_letters, f"{conf}%", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                cv2.rectangle(img_letters, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Detect words
    img_words = img.copy()
    word_boxes = pytesseract.image_to_data(img_gray, lang=lang)
    for a, b in enumerate(word_boxes.splitlines()):
        if a != 0:
            b = b.split()
            if len(b) == 12:
                x, y, w, h = int(b[6]), int(b[7]), int(b[8]), int(b[9])
                cv2.putText(img_words, b[11], (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 50, 255), 2)
                cv2.rectangle(img_words, (x, y), (x + w, y + h), (50, 50, 255), 2)

    # Close n display
    cv2.destroyAllWindows()
    cv2.imshow("Letter Confidence", img_letters)
    cv2.imshow("Word Recognition", img_words)

    print("Extracted Text:")
    print(pytesseract.image_to_string(img_gray, lang=lang))

    # Saving
    annotated_letters_path = os.path.join(os.path.expanduser("~"), "Desktop", "annotated_letters.jpg")
    annotated_words_path = os.path.join(os.path.expanduser("~"), "Desktop", "annotated_words.jpg")
    cv2.imwrite(annotated_letters_path, img_letters)
    cv2.imwrite(annotated_words_path, img_words)
    print(f"Annotated images saved at {annotated_letters_path} and {annotated_words_path}")

    # Display text
    print("Extracted text line-by-line:")
    for line in pytesseract.image_to_string(img_gray, lang=lang).splitlines():
        print(line)

# Main
def main():
    global cropped_image

    # Webcam
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        # Capture c
        if key == ord('c'):
            captured_image = frame.copy()
            cv2.imshow("Crop", captured_image)
            cv2.setMouseCallback("Crop", crop_image, captured_image)

        # Save s
        if key == ord('s'):
            if cropped_image is not None:
                desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "cropped_image.jpg")
                cv2.imwrite(desktop_path, cropped_image)
                print(f"Cropped image saved at {desktop_path}")

                process_with_tesseract(desktop_path)
            else:
                print("No cropped image to save")

        # Exit q
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
