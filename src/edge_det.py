import cv2

default_low_threshold = 100
default_high_threshold = 200

def on_trackbar_low_threshold(value):
    pass

def on_trackbar_high_threshold(value):
    pass

def display_edges_only():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cv2.namedWindow('Detected Edges')

    cv2.createTrackbar('Low Threshold', 'Detected Edges', default_low_threshold, 255, on_trackbar_low_threshold)
    cv2.createTrackbar('High Threshold', 'Detected Edges', default_high_threshold, 255, on_trackbar_high_threshold)

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        low_threshold = cv2.getTrackbarPos('Low Threshold', 'Detected Edges')
        high_threshold = cv2.getTrackbarPos('High Threshold', 'Detected Edges')

        edges = cv2.Canny(gray, low_threshold, high_threshold)
        
        # Invert the colors of the detected edges
        edges = ~edges

        cv2.imshow('Detected Edges', edges)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

display_edges_only()
