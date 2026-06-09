import base64

from IPython.display import HTML


def embed_video(path):
    with open(path, "rb") as f:
        mp4_data = base64.b64encode(f.read()).decode()
    return HTML(
        f'<video controls><source src="data:video/mp4;base64,{mp4_data}" '
        'type="video/mp4"></video>'
    )


def embed_image(path):
    with open(path, "rb") as f:
        img_data = base64.b64encode(f.read()).decode()
    return HTML(f'<img src="data:image/png;base64,{img_data}"/>')


def embed_html(path):
    with open(path, "r") as f:
        html_content = f.read()
    return HTML(html_content)
