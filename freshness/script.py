import os
import imghdr  # Standard library module for checking image type


def check_dataset_integrity(root_dir):
    print(f"Checking dataset in: {root_dir}")

    # List of known valid extensions (case-insensitive)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif')

    non_image_files = []

    for subdir, _, files in os.walk(root_dir):
        for file in files:
            file_path = os.path.join(subdir, file)

            # Check if file has a known image extension
            if not file.lower().endswith(valid_extensions):
                # Optionally, verify the file content itself
                file_type = imghdr.what(file_path)

                if file_type is None:
                    # File is neither a recognized image extension nor image content
                    non_image_files.append(file_path)

    if non_image_files:
        print("\n❌ Found non-image or invalid files. Please DELETE these:")
        for f in non_image_files:
            print(f)
        print("\nFix required: Remove the listed files and re-run your training script.")
    else:
        print("\n✅ Dataset check successful. No obvious non-image files found.")


# Run the check on your dataset folder
check_dataset_integrity(r"C:\Users\K R ARAVIND\OneDrive\Desktop\freshness\dataset")