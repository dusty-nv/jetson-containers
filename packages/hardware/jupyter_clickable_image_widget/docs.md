> This container inherits the default run command (from `jupyterlab` package) that automatically starts the Jupyter Lab.

## How to use

Open your browser, and access `http://<IP_ADDRESS>:8888`.

## How to test

Open a new Jupyter notebook, and run the following code.

```
from jupyter_clickable_image_widget import ClickableImageWidget
image_widget = ClickableImageWidget()
def on_message(_, content, ignore):
    if content['event'] == 'click':
        data = content['eventData']
        alt_key = data['altKey']
        ctrl_key = data['ctrlKey']
        shift_key = data['shiftKey']
        x = data['offsetX']
        y = data['offsetY']
image_widget.on_msg(on_message)
file = open("apple3.jpg", "rb")
image = file.read()
image_widget.value = image
display(image_widget)
```