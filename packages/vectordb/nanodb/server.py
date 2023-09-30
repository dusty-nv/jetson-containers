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
        
        # https://discuss.huggingface.co/t/how-to-serve-an-html-file/33921/2
        for n, scan in enumerate(db.scans):
            self.mounts[scan] = f"/files/{n}/"
            self.app.mount(self.mounts[scan], StaticFiles(directory=scan), name=str(n))
            
        self.create_ui()
    
    def run(self):
        # https://www.uvicorn.org/settings/
        uvicorn.run(self.app, host=self.host, port=self.port, reload=False, log_level='info')
     
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
        css = "#stats_box {font-family: monospace; font-size: 75%;} footer {visibility: hidden} body {overflow: hidden;}"
        
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
                
            gallery = gr.Gallery(
                value=self.get_random_images(self.gallery_size)
            ).style(columns=8, height='750px', object_fit='scale_down')

            gallery.select(self.on_gallery_select, None, image_upload, show_progress=False)
            text_query.change(self.on_query, text_query, [gallery, stats], show_progress=False)
            image_upload.upload(self.on_query, image_upload, [gallery, stats], show_progress=False)
            
        self.app = gr.mount_gradio_app(self.app, blocks, path='/')

    def create_stats(self, type=None):
        text = f"Model:   {self.db.model.config.name}\n"
        text += f"Images:  {len(self.db):,}\n\n"
        
        if type == 'image' and 'encode_time' in self.db.model.image_stats:
            text += f"Image Encode:  {self.db.model.image_stats.encode_time*1000:3.1f} ms\n"
        
        if type == 'text' and 'encode_time' in self.db.model.text_stats:
            text += f"Text Encode:   {self.db.model.text_stats.encode_time*1000:3.1f} ms\n"

        if 'search_time' in self.db.index.stats:
            text += f"KNN Search:    {self.db.index.stats.search_time*1000:3.1f} ms\n"
            
        return text
        #return f'<p style="display: block; float: right; text-align: right;">{txt}</p>'  #  <span style="display: block; float: right;"> vertical-align: bottom;
    
    def on_query(self, query):
        #if request:
        #    self.server_url = request.headers['origin']  # origin/referrer include http/https, 'host' does not
        #    print(f"-- server URL:  {self.server_url}")
        if isinstance(query, str):
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

        return images, gr.HTML.update(value=self.create_stats(query_type))
       
    def on_gallery_select(self, evt: gr.SelectData):
        print(f"You selected {evt.value} at {evt.index} from {evt.target}  selected={evt.selected}")
        return "/data/images/lake.jpg"
        