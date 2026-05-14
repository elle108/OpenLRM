import numpy as np
import cv2
import argparse

def colorize_image(input_path, output_path, prototxt, model, points):
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)

    # Add cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

    # Load image, scale it, convert to Lab color space
    image = cv2.imread(input_path)
    scaled = image.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_BGR2LAB)

    # Resize to 224x224 (the size the network expects), extract L channel
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50

    # Predict the 'a' and 'b' color channels
    print("Colorizing")
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))

    # Resize 'ab' back to original image size
    ab = cv2.resize(ab, (image.shape[1], image.shape[0]))

    # Grab original L channel, merge with guessed ab, convert back to BGR
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2BGR)
    colorized = np.clip(colorized, 0, 1)

    # Save
    colorized = (255 * colorized).astype("uint8")
    cv2.imwrite(output_path, colorized)
    print(f"Saved colorized image to {output_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--prototxt", default="colorization_deploy_v2.prototxt")
    ap.add_argument("--model", default="colorization_release_v2.caffemodel")
    ap.add_argument("--points", default="pts_in_hull.npy")
    args = ap.parse_args()

    colorize_image(args.input, args.output, args.prototxt, args.model, args.points)