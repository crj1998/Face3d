import gradio as gr
import cv2
from PIL import Image
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from pipeline import Face3D

app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=["detection"])
app.prepare(ctx_id=0, det_size=(640, 640))


face3d = Face3D()

def generate(source_image, target_image):
    faces = app.get(source_image)
    source_face = face_align.norm_crop(source_image, landmark=faces[0].kps, image_size=224)
    source_face = Image.fromarray(cv2.cvtColor(source_face, cv2.COLOR_BGR2RGB))
    faces = app.get(target_image)
    target_face = face_align.norm_crop(target_image, landmark=faces[0].kps, image_size=224) # you can also segment the face
    target_face = Image.fromarray(cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB))

    pred_face, pred_mask, pred_depth = face3d(source_face, target_face)
    return pred_face, pred_mask, "out.obj"

with gr.Blocks() as demo:
    gr.Markdown("Face3D WEBUI")
    with gr.Row():
        source_image = gr.Image(label="Source", type="numpy", width=512, height=512, interactive=True)
        target_image = gr.Image(label="Target", type="numpy", width=512, height=512, interactive=True)
    btn = gr.Button("Run", variant="primary")

    with gr.Row():
        model3d = gr.Model3D(label="3D", camera_position=[90.0, 90.0, 3.0], interactive=False)
        face_image = gr.Image(label="Face", type="pil", width=512, height=512, interactive=False)
        mask_image = gr.Image(label="Mask", type="pil", width=512, height=512, interactive=False)

    btn.click(fn=generate, inputs=[source_image, target_image], outputs=[face_image, mask_image, model3d])

if __name__ == "__main__":
    demo.launch(share = False)
