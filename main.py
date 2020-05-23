import cv2


def main(image_path):
    cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade.detectMultiScale(gray, scaleFactor=1.2)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255))

    cv2.imwrite('result_' + image_path, img)


if __name__ == '__main__':
    main('sample.jpg')
