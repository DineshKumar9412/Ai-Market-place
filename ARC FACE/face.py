import cv2
from mtcnn.mtcnn import MTCNN

detector = MTCNN()
img = cv2.imread("1636626913878.JPEG")
result = detector.detect_faces(img)
bounding_box = result[0]['box']

# print(bounding_box)

x1, y1, width, height = bounding_box
x2, y2 = x1 + width, y1 + height

face_boundary = img[y1:y2, x1:x2]

# img = cv2.imwrite("gfgf.jpg", face_boundary)
img = cv2.cvtColor(face_boundary, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (112, 112))

cv2.imwrite("gfgf.jpg", img)
