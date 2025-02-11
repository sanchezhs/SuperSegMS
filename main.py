from ultralytics import YOLO

model = YOLO("yolo11n.pt")  # initialize model
results = model("./dog.jpg")  # perform inference

results[0].show()  # display results for the first image
results[0].plot()  # display results for the first image
