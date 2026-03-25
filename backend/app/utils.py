import base64
import io
from PIL import Image


def load_image_from_base64(encoded: str) -> Image.Image:
    header, _, data = encoded.partition(',')
    if not data:
        data = header
    image_data = base64.b64decode(data)
    return Image.open(io.BytesIO(image_data)).convert('RGB')
