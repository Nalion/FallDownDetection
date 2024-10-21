from roboflow import Roboflow

rf = Roboflow(api_key="lGCm2PEaF2UV3RZy67fW")
project = rf.workspace("yolo-7gnh8").project("yolov5-obvge")
version = project.version(1)
dataset = version.download("yolov5")
