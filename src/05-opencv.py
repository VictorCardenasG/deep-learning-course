import cv2
def display_webcam():
    # Open a connection to the webcam (0 is the default ID for the primary camera)
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        # Capture frame-by-frame
        ret, frame=cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        # Display the resulting frame
        cv2.imshow('Webcam', frame)

        # Press 'q' on the keyboard to exit the webcam feed
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    # When everything is done, release the capture and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    display_webcam()