from ultralytics import YOLO

# Load the custom trained model
print("Loading Custom YOLO11 Large model...")
model = YOLO(r"D:\OmniDrive\api\models\car_parts_large_v1.pt")

# Path to the specific image
image_path = r"D:\OmniDrive\dataset1\merged\train\AIR COMPRESSOR\50Car__train__001.jpg"

print(f"\nPredicting image: {image_path}")

# Run prediction
results = model(image_path)

# Print out the results securely
result = results[0]
top5 = result.probs.top5
confs = result.probs.top5conf

print("\n--- PREDICTION RESULTS ---")
for idx, conf in zip(top5, confs):
    class_name = result.names[idx]
    confidence = float(conf) * 100
    print(f"{class_name}: {confidence:.2f}%")
print("--------------------------\n")
