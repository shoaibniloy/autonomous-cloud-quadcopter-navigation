from ultralytics import YOLO

# Load the model
model = YOLO("furnitureepoch50.pt")

# Assign to global variable
CLASS_MAPPINGS = model.names

# Print the class mappings
print("Class Mappings:")
for class_id, class_name in CLASS_MAPPINGS.items():
    print(f"{class_id}: {class_name}")
