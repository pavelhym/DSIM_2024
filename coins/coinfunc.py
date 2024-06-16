import matplotlib.pyplot as plt


SHOW_PLOTS = True


def show_plot():
    if SHOW_PLOTS:
        st.pyplot()

def detect_yolo(image_path,model, streamlit = True):
    image_path = image_path 
    image = cv2.imread(image_path)
    
    # Perform object detection
    results = model(image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the predictions
    boxes = results[0].boxes  # Get the detected boxes
    for box in boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()  # Get the coordinates of the box
        label = box.cls.item()  # Get the class label
        confidence = box.conf.item()  # Get the confidence score as a Python float
        # Draw the box and label on the image
        cv2.rectangle(image_rgb, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(image_rgb, f'{model.names[int(label)]} {confidence:.2f}', (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    plt.imshow(image_rgb)
    plt.axis('off')
    if streamlit:
        show_plot()
    else:
        plt.show()

    