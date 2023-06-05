from hydra import compose, initialize
from omegaconf import OmegaConf
import torchvision
import numpy as np
import gradio as gr
# import cv2 

from facetorch import FaceAnalyzer

from db import VectorDB

# init pipeline
initialize(config_path="../conf", job_name="test_app")
detector_cfg = compose(config_name="liveness_detector")
extractor_cfg = compose(config_name="feat_extractor.yaml")

detector = FaceAnalyzer(detector_cfg.analyzer)
extractor = FaceAnalyzer(extractor_cfg.analyzer)

vector_db = VectorDB("liveness_detection_prod")
vector_db.create_collection(embedding_size=512)

def upload_selfie_image(image, user_name, sess):  
    sess["user_name"] = user_name  
    output = extractor.run(image=image, include_tensors=True)
    
    if(len(output.faces) != 1):
        return sess, "Image must contain one face"
    
    face_embed = output.faces[0].preds["verify"].logits.detach().cpu().tolist()
    
    result = vector_db.add_face_emb(face_embed, user_name)
    return sess, result

def image_demo_handler(image, sess):
    output = detector.run(image=image.copy(), return_img_data=True, include_tensors=True)
    
    # get vis img
    vis_img = torchvision.transforms.functional.to_pil_image(output.img)
        
    if(len(output.faces) != 1):
        return None, "Image must contain one face"
    
    # verify user 
    face_embed = output.faces[0].preds["verify"].logits.detach().cpu().tolist()
    user_name = sess["user_name"]
    if(vector_db.verify_user(face_embed, user_name, threshold=0.5)):
        bbox = output.faces[0].loc
        asf_out = output.faces[0].preds["antispoof"]
        
        if(asf_out.label == "Real"):
            return vis_img, f"user <{user_name}> is liveness"
        else:
            return vis_img, f"user <{user_name}> is not liveness ({asf_out.label} attack)"
        
    else:
        return vis_img, f"You are not look like {user_name}"

# def video_demo_handler(video): 
#     cap = cv2.VideoCapture(video)
    
#     input_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#     input_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     input_fps = cap.get(cv2.CAP_PROP_FPS)
    
#     output_file = "./test.webm"
#     writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'VP90'), 15, (int(input_width), int(input_height))) 
    
#     if (cap.isOpened()== False): 
#         print("Error opening video stream or file")
 
#     # Read until video is completed
#     while(cap.isOpened()):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             break 
        
#         output_image = detector.run(frame)
        
#         writer.write(output_image)
            
#     cap.release()
#     writer.release()
    
#     return output_file

def webcam_handler(image, sess):
    output = detector.run(image=image.copy(), return_img_data=False, include_tensors=True)
    
    if(len(output.faces) != 1):
        return None, "Image must contain one face"
    
    # verify user 
    face_embed = output.faces[0].preds["verify"].logits.detach().cpu().tolist()
    user_name = sess["user_name"]
    if(vector_db.verify_user(face_embed, user_name, threshold=0.5)):
        bbox = output.faces[0].loc
        asf_out = output.faces[0].preds["antispoof"]
        
        if(asf_out.label == "Real"):
            return f"user <{user_name}> is liveness"
        else:
            return f"user <{user_name}> is not liveness ({asf_out.label} attack)"
        
    else:
        return f"You are not look like {user_name}"

with gr.Blocks() as demo:
    gr.Markdown("Try our face liveness detection")
    sess = gr.State(value={"user_name": "unknown"})
    print(sess.value)
    
    with gr.Tab("Upload selfie image"): 
        input_image = gr.Image(type="pil")
        user_name = gr.Textbox(label="User Name")
        text_output = gr.Textbox(label="Output Messages")
        
        upload_selfie_button = gr.Button("Upload")
        upload_selfie_button.click(fn=upload_selfie_image, inputs=[input_image, user_name, sess], outputs=[sess, text_output])
        
    with gr.Tab("Image Demo"):
        with gr.Row():
            demo_image_input = gr.Image(label="Input image", type="pil")
            demo_image_output = gr.Image(lable="Output image")
            
        demo_image_text_output = gr.Textbox(label="Liveness detection result")
        demo_image_button = gr.Button("Upload Image")
        
        demo_image_button.click(fn=image_demo_handler, inputs=[demo_image_input, sess], outputs=[demo_image_output, demo_image_text_output])
        
    # with gr.Tab("Video Demo"):
    #     with gr.Row():
    #         demo_video_input = gr.Video(label="Video")
    #         demo_video_output = gr.Textbox(label="Liveness detection result")
    #     demo_video_button = gr.Button("Upload Video")
        
    #     demo_video_button.click(fn=video_demo_handler, inputs=demo_video_input, outputs=demo_video_output)
        
    with gr.Tab("Webcam Demo"):
        with gr.Row():
            webcam_input = gr.Image(source="webcam", streaming=True)
            webcam_output = gr.Textbox(label="Liveness detection result")
            webcam_input.change(webcam_handler,
                           inputs=[webcam_input, sess],
                           outputs=webcam_output)
    
if __name__ == "__main__":
    demo.launch(show_error=True, share=False, server_name="0.0.0.0")
    
