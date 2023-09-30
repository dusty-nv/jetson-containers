#!/usr/bin/env python3
import os
import PIL
import pprint
import socket
import threading
import numpy as np

import fastapi
import uvicorn
import gradio as gr

from fastapi.staticfiles import StaticFiles


class Server(threading.Thread):
    def __init__(self, db, host='0.0.0.0', port=7860):
        super(Server, self).__init__(daemon=True)  # stop thread on main() exit

        self.db = db
        self.app = fastapi.FastAPI()
        self.host = host
        self.port = port
        self.mounts = {}
        self.server_url = f"http://{socket.gethostbyname(socket.gethostname())}:{port}"
        
        self.gallery_size = 64
        self.gallery_images = None
        
        # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
        for n, scan in enumerate(db.scans):
            self.mounts[scan] = f"/files/{n}/"
            self.app.mount(self.mounts[scan], StaticFiles(directory=scan), name=str(n))
            
        self.create_ui()
    
    def run(self):
        # https://www.uvicorn.org/settings/
        uvicorn.run(self.app, host=self.host, port=self.port, reload=False, log_level='warning')  # 'info'
     
    def get_random_images(self, n):
        indexes = np.random.randint(0, len(self.db)-1, n)
        images = []
        
        for m in range(n):
            path = self.db.metadata[indexes[m]]['path']
            """
            for scan_dir, mount_dir in self.mounts.items():
                print('scan_dir', scan_dir, 'mount_dir', mount_dir)
                path = path.replace(scan_dir, mount_dir)
            images.append(self.server_url + path)
            """
            images.append(path)

        return images
        
    def create_ui(self):
        css = """
            #stats_box {font-family: monospace; font-size: 65%; height: 162px;} 
            footer {visibility: hidden} 
            body {overflow: hidden;}
            * {scrollbar-color: rebeccapurple green; scrollbar-width: thin;}
        """
        # https://stackoverflow.com/questions/66738872/why-doesnt-the-scrollbar-color-property-work-directly-on-the-body

        with gr.Blocks(css=css, theme=gr.themes.Monochrome()) as blocks:
            gr.HTML('<h1 style="color: #6aa84f; font-size: 250%;">nanodb</h1>')
            
            #with gr.Row() as row:
                #gr.HTML(f'<p style="display: block; float: right; text-align: right;">Put right side text here </br> abc 123</p>')
            
            with gr.Row().style(equal_height=True):
                with gr.Column():
                    text_query = gr.Textbox(placeholder="Search Query", show_label=False)#.style(container=False)
                    stats = gr.Textbox(
                        value=self.create_stats(type), 
                        lines=5, 
                        show_label=False, 
                        interactive=False, 
                        elem_id='stats_box'
                    )
                        
                image_upload = gr.Image(type='pil')
            
            self.gallery_images = self.get_random_images(self.gallery_size)
            
            gallery = gr.Gallery(
                value=self.gallery_images
            ).style(columns=8, height='750px', object_fit='scale_down', preview=False)

            gallery.select(self.on_gallery_select, None, [gallery, stats, image_upload, text_query], show_progress=False)
            text_query.change(self.on_query, text_query, [gallery, stats, image_upload], show_progress=False)
            image_upload.upload(self.on_query, image_upload, [gallery, stats, text_query], show_progress=False)
            
        self.app = gr.mount_gradio_app(self.app, blocks, path='/')

    def create_stats(self, type=None):
        text = f"Model:   CLIP {self.db.model.config.name}\n"
        text += f"Images:  {len(self.db):,}\n\n"
        
        if type == 'image' and 'encode_time' in self.db.model.image_stats:
            text += f"Image Encode:  {self.db.model.image_stats.encode_time*1000:3.1f} ms\n"
        
        if type == 'text' and 'encode_time' in self.db.model.text_stats:
            text += f"Text Encode:   {self.db.model.text_stats.encode_time*1000:3.1f} ms\n"

        if 'search_time' in self.db.index.stats:
            text += f"KNN Search:    {self.db.index.stats.search_time*1000:3.1f} ms"
            
        return text
        #return f'<p style="display: block; float: right; text-align: right;">{txt}</p>'  #  <span style="display: block; float: right;"> vertical-align: bottom;
    
    def on_query(self, query):
        #if request:
        #    self.server_url = request.headers['origin']  # origin/referrer include http/https, 'host' does not
        #    print(f"-- server URL:  {self.server_url}")
        if isinstance(query, str):
            if os.path.splitext(query)[1].lower() in self.db.img_extensions:
                print(f"-- image query from path {query}")
                query_type='image'
            else:
                print(f"-- web text query '{query}'")
                query_type='text'
        elif isinstance(query, PIL.Image.Image):
            print(f"-- web image query  {query.size}  {type(query)}")
            query_type='image'
        else:
            raise ValueError(f"unexpected query type {type(query)}")
            
        indexes, distances = self.db.search(query, k=self.gallery_size)
        images = []
        
        for n in range(self.gallery_size):
            images.append((self.db.metadata[indexes[n]]['path'], f"{distances[n]*100:.1f}%"))

        self.gallery_images = images
        return images, gr.HTML.update(value=self.create_stats(query_type)), None
       
    def on_gallery_select(self, evt: gr.SelectData):
        print(f"-- user selected {evt.value} at {evt.index} from {evt.target}  selected={evt.selected}")
        if evt.index < len(self.gallery_images):
            img = self.gallery_images[evt.index]
            if not isinstance(img, str):
                img = img[0]
        else:
            img = "/data/images/lake.jpg"
          
        images, stats, _ = self.on_query(img)
        return images, stats, img, None
        
        