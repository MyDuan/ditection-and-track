from yolov4.tf import YOLOv4

yolo = YOLOv4()
yolo.classes = "./yolofile/coco.names"
yolo.make_model()
yolo.load_weights("./yolofile/yolov4.weights", weights_type="yolo")

yolo.inference(media_path="./sample_data/pigandperson.jpg")
