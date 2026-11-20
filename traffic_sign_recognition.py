   import argparse
   from pathlib import Path

   import numpy as np
   import tensorflow as tf
   from PIL import Image

   IMAGE_SIZE = (32, 32)


   def load_image(image_path: str) -> np.ndarray:
       path = Path(image_path)
       if not path.is_file():
           raise FileNotFoundError(f"Image not found: {path}")

       image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
       return np.asarray(image, dtype=np.float32) / 255.0


   def main() -> None:
       parser = argparse.ArgumentParser(
           description="Classify a traffic-sign image with a trained Keras model."
       )
       parser.add_argument("--image", required=True, help="Path to an input image")
       parser.add_argument(
           "--model",
           default="models/traffic_sign_model.keras",
           help="Path to a trained Keras model",
       )
       args = parser.parse_args()

       model_path = Path(args.model)
       if not model_path.is_file():
           raise FileNotFoundError(
               f"Model not found: {model_path}. Add a trained model before prediction."
           )

       model = tf.keras.models.load_model(model_path)
       image = load_image(args.image)
       probabilities = model.predict(np.expand_dims(image, axis=0), verbose=0)[0]
       class_id = int(np.argmax(probabilities))

       print(f"Predicted class ID: {class_id}")
       print(f"Confidence: {float(probabilities[class_id]):.2%}")


   if __name__ == "__main__":
       main()
