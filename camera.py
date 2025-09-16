import cv2
import numpy as np
import pytesseract
import imutils
import time

def preprocess_plate(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = max(1, 400 // w)
    gray = cv2.resize(gray, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY, 11, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=1)
    return th

def ocr_with_confidence(img):
    config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    data = pytesseract.image_to_data(img, config=config, output_type=pytesseract.Output.DICT)
    
    best_text = ""
    best_conf = 0
    
    for i, word in enumerate(data["text"]):
        if word.strip() != "":
            conf = int(data["conf"][i]) if data["conf"][i].isdigit() else 0
            if conf > best_conf:
                best_conf = conf
                best_text = ''.join(ch for ch in word if ch.isalnum() or ch == '-')
    
    return best_text, best_conf

def find_plate_candidates(frame):
    img = imutils.resize(frame, width=800)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    edged = cv2.Canny(gray, 30, 200)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:50]
    candidates = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = w / float(h) if h > 0 else 0
            area = w * h
            if 2.0 <= aspect_ratio <= 6.5 and area > 1000:
                scale = frame.shape[1] / 800.0
                rx, ry, rw, rh = int(x*scale), int(y*scale), int(w*scale), int(h*scale)
                candidates.append((rx, ry, rw, rh))
    return candidates

def main(camera_index=0):
    cap = cv2.VideoCapture(camera_index)
    time.sleep(1.0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        candidates = find_plate_candidates(frame)
        for (x, y, w, h) in candidates:
            plate_crop = frame[y:y+h, x:x+w]
            if plate_crop.size == 0:
                continue

            prep = preprocess_plate(plate_crop)
            text, conf = ocr_with_confidence(prep)

            if text and conf >= 70:  # Only accept if confidence ≥ 70%
                cv2.putText(frame, f"{text} ({conf}%)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                print(f"Detected Plate: {text}, Confidence: {conf}%")

        # show live camera feed
        cv2.imshow("ANPR Camera", frame)

        # press 'q' to quit manually
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


[fuel type, vehicle type,engine capacity, transmissions, Fuel Consumption Comb (L/100 km), vehicle_age, Fuel_Consumed_Last_Month, ]