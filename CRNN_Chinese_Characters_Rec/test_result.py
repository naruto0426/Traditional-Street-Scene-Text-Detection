
import torch,cv2
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).cuda()
path = "../train/img/img_1.jpg"
frame = cv2.imread(path)
results = model([frame])
labels, cords = results.xyxyn[0][:, -1].cpu().numpy(), results.xyxyn[0][:, :-1].cpu().numpy()
x_shape, y_shape = frame.shape[1], frame.shape[0]
n = len(labels)
for i in range(n):
    row = cords[i]
    x1, y1, x2, y2 = int(row[0]*x_shape), int(row[1]*y_shape), int(row[2]*x_shape), int(row[3]*y_shape)
    bgr = (0, 0, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), bgr, 1)
    label = f"{int(row[4]*100)}"
    cv2.putText(frame, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    cv2.putText(frame, f"Total Targets: {n}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
cv2.imwrite("result.jpg",frame)
