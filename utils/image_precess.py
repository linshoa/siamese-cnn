import warnings
import numpy as np
import PIL
import cv2

"""reference https://github.com/keras-team/keras"""
try:
    from PIL import Image as pil_image
except ImportError:
    raise ImportError('Could not import PIL.Image')

if pil_image:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
        'bilinear': pil_image.BILINEAR,
        'bicubic': pil_image.BICUBIC
    }
    # These methods were only introduced in version 3.4.0 (2016).
    if hasattr(pil_image, 'HAMMING'):
        _PIL_INTERPOLATION_METHODS['hamming'] = pil_image.HAMMING
    if hasattr(pil_image, 'BOX'):
        _PIL_INTERPOLATION_METHODS['box'] = pil_image.BOX
    # This method is new in version 1.1.3 (2013).
    if hasattr(pil_image, 'LANCZOS'):
        _PIL_INTERPOLATION_METHODS['lanczos'] = pil_image.LANCZOS


def img_2_array(img, data_format='channels_last'):
    """Converts a PIL Image instance to a Numpy array.

    # since tf only accept channels_last, we only accept that!

    # Arguments
        img: PIL Image instance.
        data_format: Image data format,
            either "channels_first" or "channels_last".

    # Returns
        A 3D Numpy array.

    # Raises
        ValueError: if invalid `img` or `data_format` is passed.
    """
    if data_format not in {'channels_last'}:
        raise ValueError('Only channels_last accept, but unknown data_format: ', data_format)

    # Numpy array x has format (height, width, channel)
    # but original PIL image has format (width, height, channel)
    # numpy.asarray convert the input to array
    array = np.asarray(img, dtype="float32")
    print(array.shape)
    if len(array.shape) >= 2:
        if len(array.shape) == 2:
            array = array.reshape(array[0], array[1], 1)
    else:
        raise ValueError('Unsupported image shape: ', array.shape)
    return array


def array_to_img(x, data_format='channels_last', scale=True):
    """Converts a 3D Numpy array to a PIL Image instance.

    # Arguments
        x: Input Numpy array.
        data_format: Image data format.
            either "channels_first" or "channels_last".
        scale: Whether to rescale image values
            to be within `[0, 255]`.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if invalid `x` or `data_format` is passed.
    """
    x = np.asarray(x, dtype="float32")
    if x.ndim != 3:
        raise ValueError('Expected image array to have rank 3 (single image). '
                         'Got array with shape:', x.shape)
    if data_format not in {'channels_last'}:
        raise ValueError('Invalid data_format:', data_format)

    # Original Numpy array x has format (height, width, channel)
    # or (channel, height, width)
    # but target PIL image has format (width, height, channel)
    # if data_format == 'channels_first':
    #     x = x.transpose(1, 2, 0)
    if scale:
        x = x + max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 4:
        # RGBA
        return pil_image.fromarray(x.astype('uint8'), 'RGBA')
    elif x.shape[2] == 3:
        # RGB
        return pil_image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return pil_image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise ValueError('Unsupported channel number: ', x.shape[2])


def load_img(path, grayscale=False, color_mode='rgb', target_size=None,
             interpolation='nearest'):
    """Loads an image into PIL format.

    # Arguments
        path: Path to image file.
        color_mode: One of "grayscale", "rbg", "rgba". Default: "rgb".
            The desired image format.
        target_size: Either `None` (default to original size)
            or tuple of ints `(img_height, img_width)`.
        interpolation: Interpolation method used to resample the image if the
            target size is different from that of the loaded image.
            Supported methods are "nearest", "bilinear", and "bicubic".
            If PIL version 1.1.3 or newer is installed, "lanczos" is also
            supported. If PIL version 3.4.0 or newer is installed, "box" and
            "hamming" are also supported. By default, "nearest" is used.

    # Returns
        A PIL Image instance.

    # Raises
        ImportError: if PIL is not available.
        ValueError: if interpolation method is not supported.
    """
    if grayscale is True:
        warnings.warn('grayscale is deprecated. Please use '
                      'color_mode = "grayscale"')
        color_mode = 'grayscale'

    img = pil_image.open(path)
    if color_mode == 'grayscale':
        if img.mode != 'L':
            img = img.convert('L')
    elif color_mode == 'rgba':
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
    elif color_mode == 'rgb':
        if img.mode != 'RGB':
            img = img.convert('RGB')
    else:
        raise ValueError('color_mode must be "grayscale", "rbg", or "rgba"')
    if target_size is not None:
        width_height_tuple = (target_size[1], target_size[0])
        if img.size != width_height_tuple:
            if interpolation not in _PIL_INTERPOLATION_METHODS:
                raise ValueError(
                    'Invalid interpolation method {} specified. Supported '
                    'methods are {}'.format(
                        interpolation,
                        ", ".join(_PIL_INTERPOLATION_METHODS.keys())))
            resample = _PIL_INTERPOLATION_METHODS[interpolation]
            img = img.resize(width_height_tuple, resample)
    return img


def preprocess_input(x, data_format='channels_last', mode='tf'):
    """Preprocesses a tensor or Numpy array encoding a batch of images.

        # Arguments
            x: Input Numpy or symbolic tensor, 3D or 4D.
                The preprocessed data is written over the input data
                if the data types are compatible. To avoid this
                behaviour, `numpy.copy(x)` can be used.
            data_format: Data format of the image tensor/array.
            mode: One of "caffe", "tf" or "torch".
                - caffe: will convert the images from RGB to BGR,
                    then will zero-center each color channel with
                    respect to the ImageNet dataset,
                    without scaling.
                - tf: will scale pixels between -1 and 1,
                    sample-wise.
                - torch: will scale pixels between 0 and 1 and then
                    will normalize each channel with respect to the
                    ImageNet dataset.

        # Returns
            Preprocessed tensor or Numpy array.

        # Raises
            ValueError: In case of unknown `data_format` argument.
        """
    if isinstance(x, np.ndarray):
        if not issubclass(x.dtype.type, np.floating):
            x = x.astype('float32', copy=False)

        if mode == 'tf':
            x /= 127.5
            x -= 1
            return x 
    else:
        raise ValueError('the format of input is not np.ndarray')


# if __name__ == '__main__':
#     file_directory = '/home/ubuntu/media/File/1Various/Person_reid_dataset/Market-1501-v15.09.15/'
#     img_dir = 'query/0014_c5s1_001026_00.jpg'
#
#     img = load_img(file_directory+img_dir, target_size=[224, 224], interpolation='bilinear')
#
#     img = img_2_array(img)
#     x = preprocess_input(img)
#     print(x)
#     # shape (224, 224, 3)

    # too slow !!!
    # image = cv2.imread(file_directory+img_dir)
    # image = cv2.resize(image, (224, 224))
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # print(image.shape)


