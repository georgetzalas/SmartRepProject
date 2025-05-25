import os
import json
import fitz  # This is PyMuPDF


def extract_images_with_page_numbers(pdf_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)

    images_by_page = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_index, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            image_filename = f"page{page_num + 1}img{img_index + 1}.{image_ext}"
            image_path = os.path.join(output_dir, image_filename)

            with open(image_path, "wb") as f:
                f.write(image_bytes)

            if (page_num + 1) not in images_by_page:
                images_by_page[page_num + 1] = []
            images_by_page[page_num + 1].append(image_path)

        with open('image_database.json', 'w') as f:
            json.dump(images_by_page, f)

# extract_images_with_page_numbers("../data/bmw_x1.pdf", "../data/images")
