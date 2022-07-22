import os, time
from traceback import print_exc
from urllib.request import urlopen
from io import BytesIO
from PIL import Image, ImageFile


def call_with_retry(fn, *args, _max_retries=5, _sleep_time=1, _verbose=False, **kwargs):
    for retry in range(_max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if retry == _max_retries - 1:
                raise e
            if _verbose:
                print_exc()
            time.sleep(_sleep_time)


def open_image(image_url=None, image_path=None, bin_image=None):
    assert (image_url and not image_path and not bin_image) or \
        (not image_url and not bin_image and image_path and os.path.isfile(image_path)) or \
        (not image_url and not image_path and bin_image), \
        'Invalid input argument'
    content = urlopen(image_url).read() if image_url \
        else (open(image_path, 'rb').read() if image_path else bin_image)
    image = Image.open(BytesIO(content)).convert('RGB')
    image = _apply_exif_orientation(image)
    return image


# To resolve 'OSError: broken data stream when reading image file'
ImageFile.LOAD_TRUNCATED_IMAGES = True

# https://www.exiv2.org/tags.html
_EXIF_ORIENT = 274  # exif 'Orientation' tag


def _apply_exif_orientation(image):
    """
    Applies the exif orientation correctly.
    This code exists per the bug:
      https://github.com/python-pillow/Pillow/issues/3973
    with the function `ImageOps.exif_transpose`. The Pillow source raises errors with
    various methods, especially `tobytes`
    Function based on:
      https://github.com/wkentaro/labelme/blob/v4.5.4/labelme/utils/image.py#L59
      https://github.com/python-pillow/Pillow/blob/7.1.2/src/PIL/ImageOps.py#L527
    Args:
        image (PIL.Image): a PIL image
    Returns:
        (PIL.Image): the PIL image with exif orientation applied, if applicable
    """
    if not hasattr(image, "getexif"):
        return image

    try:
        exif = image.getexif()
    except Exception:  # https://github.com/facebookresearch/detectron2/issues/1885
        exif = None

    if exif is None:
        return image

    orientation = exif.get(_EXIF_ORIENT)

    method = {
        2: Image.FLIP_LEFT_RIGHT,
        3: Image.ROTATE_180,
        4: Image.FLIP_TOP_BOTTOM,
        5: Image.TRANSPOSE,
        6: Image.ROTATE_270,
        7: Image.TRANSVERSE,
        8: Image.ROTATE_90,
    }.get(orientation)

    if method is not None:
        return image.transpose(method)
    return image
