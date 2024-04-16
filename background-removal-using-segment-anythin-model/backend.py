from segment_anything import SamPredictor, sam_model_registry
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import base64

# create sam predictor
model_path = 'sam_vit_b_01ec64.pth'
if not os.path.exists(model_path):
    model_path = 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth'

sam = sam_model_registry["vit_b"](checkpoint=model_path)
predictor = SamPredictor(sam)

# load image and select x, y coordinates to test
image_path = 'data/test.jpg'

x = 985
y = 518

image = cv2.imread(image_path)

_, image_bytes = cv2.imencode('.png', image)

image_bytes = image_bytes.tobytes()

image_bytes_encoded_base64 = base64.b64encode(image_bytes).decode('utf-8')

# wrap it up as a function
def remove_background(image_base64_encoding, x, y):
    
    image_bytes = base64.b64decode(image_base64_encoding)
    
    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    predictor.set_image(image)
    
    masks, scores, logits = predictor.predict(
                                        point_coords = np.asarray([[x,y]]),
                                        point_labels = np.asarray([1]),
                                        multimask_output=True
                                        )
    C, H, W = masks.shape
    result_mask = np.zeros((H,W), dtype=bool)
    
    for j in range(C):
        result_mask |= masks[j,:,:]   
        
    result_mask = result_mask.astype(np.uint8)

    alpha_channel = np.ones(result_mask.shape, dtype=result_mask.dtype) * 255
    
    alpha_channel[result_mask == 0] = 0
    
    result_image = cv2.merge((image, alpha_channel))
    
    _, result_image_bytes = cv2.imencode('.png', result_image)
    
    result_image_bytes = result_image_bytes.tobytes()
    
    result_image_bytes_encoded_base64 = base64.b64encode(result_image_bytes).decode('utf-8')
    
    return result_image_bytes_encoded_base64

result_image = remove_background(image_bytes_encoded_base64, x, y)

result_image_bytes = base64.b64decode(result_image)

result_image = cv2.imdecode(np.frombuffer(result_image_bytes, dtype=np.uint8), cv2.IMREAD_UNCHANGED)

plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGRA2RGBA))
plt.show()


