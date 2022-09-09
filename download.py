import labelbox
# Enter your Labelbox API key here
from labelbox.data.serialization import COCOConverter

if __name__ == '__main__':
    LB_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbDZrbXF6ZmE1anUxMDd4bGJ3NGVneThwIiwib3JnYW5pemF0aW9uSWQiOiJjbDZrbXF6ZWo1anR4MDd4bDd3NzU1eWs2IiwiYXBpS2V5SWQiOiJjbDc4NDVpY2kycWUyMDd4azNrdTI0MWw3Iiwic2VjcmV0IjoiNDUzYzU5NTIzYmU4YmRiZDRkYmQ3M2QzMDYwYjA4OTEiLCJpYXQiOjE2NjEzNzU2MjcsImV4cCI6MjI5MjUyNzYyN30.9PXvIHriixS9_C8invRr3xMz178aqqeulCK4gM-LdDs"
    # Create Labelbox client
    lb = labelbox.Client(api_key=LB_API_KEY)
    # Get project by ID
    project = lb.get_project('cl71omjf61lip08zm5am04blh')
    # Export image and text data as an annotation generator:
    labels = project.label_generator()


    mask_path = "./masks/"
    image_path = './images/'

    coco_labels = COCOConverter.serialize_panoptic(
        labels,
        image_root=image_path,
        mask_root=mask_path,
        ignore_existing_data=True
    )