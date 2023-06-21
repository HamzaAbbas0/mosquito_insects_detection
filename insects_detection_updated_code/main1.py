import cv2
import os

def detect_insects(image_folder, output_folder, scale_factor=1.5):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get the list of image files in the folder
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.jpg') or f.endswith('.jpeg') or f.endswith('.png')]

    # Store the counts of extracted pics for each original photo
    extraction_counts = {}

    # Iterate over each image file
    for image_file in image_files:
        # Load the image
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Save the grayscale image
        gray_image_path = os.path.join(output_folder, f"gray_{image_file}")
        cv2.imwrite(gray_image_path, gray)
        # Print the grayscale image
        # cv2.imshow("Grayscale Image", gray)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply a Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Save the blurred image
        blurred_image_path = os.path.join(output_folder, f"blurred_{image_file}")
        cv2.imwrite(blurred_image_path, blurred)
        # Print the blurred image
        # cv2.imshow("Blurred Image", blurred)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Apply Canny edge detection to find edges
        edges = cv2.Canny(blurred, 50, 150)

        # Save the edges image
        edges_image_path = os.path.join(output_folder, f"edges_{image_file}")
        cv2.imwrite(edges_image_path, edges)
        # Print the edges image
        # cv2.imshow("Edges Image", edges)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


        # Find contours in the image
        if cv2.__version__.startswith('4'):
            contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        else:
            _, contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Create a folder for the current image's extracted photos
        image_name = os.path.splitext(image_file)[0]
        output_image_folder = os.path.join(output_folder, image_name)
        os.makedirs(output_image_folder, exist_ok=True)

        # Iterate over the contours and save each insect
        extraction_count = 0
        for i, contour in enumerate(contours):
            # Filter out small contours
            if cv2.contourArea(contour) > 100:
                # Extract the insect from the original image
                x, y, w, h = cv2.boundingRect(contour)

                # Increase the width and height of the bounding rectangle
                new_width = int(w * scale_factor)
                new_height = int(h * scale_factor)
                x -= int((new_width - w) / 2)
                y -= int((new_height - h) / 2)
                w = new_width
                h = new_height

                # Adjust the coordinates to ensure they are within the image boundaries
                x = max(0, x)
                y = max(0, y)
                x_end = min(x + w, image.shape[1])
                y_end = min(y + h, image.shape[0])
                w = x_end - x
                h = y_end - y

                insect = image[y:y+h, x:x+w]

                # Save the insect image
                insect_filename = f'{image_name}-insect_{extraction_count}.jpg'
                cv2.imwrite(os.path.join(output_image_folder, insect_filename), insect)

                extraction_count += 1

        # Store the extraction count for the current image
        extraction_counts[image_name] = extraction_count

    # Generate the report
    report_filename = os.path.join(output_folder, 'extraction_report.txt')
    with open(report_filename, 'w') as report_file:
        for image_name, extraction_count in extraction_counts.items():
            report_file.write(f"Label of the originating photo {image_name}: {extraction_count} extracted pics.\n")

        total_extracted = sum(extraction_counts.values())
        report_file.write(f"\nTotal pics extracted: {total_extracted}.")
        print("output is generated successfully ")

# Example usage
image_folder = 'image_folder'
output_folder = 'extracted_output_Result/'
scale_factor = 1.5  # Adjust the scale factor as desired
detect_insects(image_folder, output_folder, scale_factor)
