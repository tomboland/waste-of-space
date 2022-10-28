import sys
import cv2
import pdf2image
import img2pdf
from PIL.Image import Image
import numpy as np
from funcy import first, last
import poppler
from returns.context import Reader
from dataclasses import dataclass
from typing import Callable, Tuple


@dataclass
class _Cv2Settings():
    max_value: int
    adaptive_method: int
    threshold_type: int
    block_size: int
    adaptive_constant: int
    kernel_size: int
    edge_crop_size: int
    color_format: int


def pdf_page_to_pillow_image(file_bytes: bytes, page: int) -> Image:
    return first(pdf2image.convert_from_bytes(
        file_bytes, dpi=200, output_folder=None, first_page=page,
        last_page=page, fmt='jpg', thread_count=1, userpw=None,
        use_cropbox=False, strict=False
    ))


def pillow_image_to_cv2(image: Image) -> np.ndarray:
    return np.asarray(image)


def image_to_greyscale(image: np.ndarray) -> Reader[np.ndarray, _Cv2Settings]:
    return Reader(
        lambda settings: cv2.cvtColor(image, settings.color_format)
    )


def invert_image(image: np.ndarray) -> Reader[np.ndarray, _Cv2Settings]:
    return Reader(
        lambda _: cv2.bitwise_not(image)
    )


def image_adaptive_threshold(image: np.ndarray) -> Reader[np.ndarray, _Cv2Settings]:
    return Reader(
        lambda settings: cv2.adaptiveThreshold(
            src=image,
            maxValue=settings.max_value,
            adaptiveMethod=settings.adaptive_method,
            thresholdType=settings.threshold_type,
            blockSize=settings.block_size,
            C=settings.adaptive_constant
        )
    )


def erode_image(image: np.ndarray) -> Reader[np.ndarray, _Cv2Settings]:
    return Reader(
        lambda settings: cv2.erode(
            image,
            np.ones(
                [settings.kernel_size, settings.kernel_size], np.uint8
            ),
            iterations=2
        )
    )


def sharpen_image(image: np.ndarray) -> np.ndarray:
    sharpening_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, sharpening_filter)


def bilateral_filter(image: np.ndarray) -> np.ndarray:
    return cv2.bilateralFilter(image, 9, 75, 75)


def denoise(image: np.ndarray) -> np.ndarray:
    return cv2.fastNlMeansDenoising(image, None, 20, 7, 21)


def threshold(image: np.ndarray) -> np.ndarray:
    ret, img = cv2.threshold(image, 1, 255, cv2.THRESH_BINARY)
    return img


def get_image_cropper(image: np.ndarray) -> Callable[[Tuple[int, int, int, int]], np.ndarray]:
    def _crop_image(xy: Tuple[int, int, int, int]) -> np.ndarray:
        return image[xy[0]:xy[1], xy[2]:xy[3]]
    return _crop_image


def find_margin_extents(
    image: np.ndarray,
    black_threshold=150,
    threshold_pc: float = 2
) -> Tuple[int, int, int, int]:
    height, width = image.shape
    over_threshold = image < black_threshold
    column_sums = np.where(over_threshold.sum(axis=1) >= (threshold_pc / 100.0 * width))[0]
    row_sums = np.where(over_threshold.sum(axis=0) >= (threshold_pc / 100.0 * height))[0]
    left, right = first(column_sums) or 0, last(column_sums) or width
    top, bottom = first(row_sums) or 0, last(row_sums) or height
    cv2.imwrite("cropped.jpg", image[left:right, top:bottom])
    return (left, right, top, bottom)


if __name__ == "__main__":

    image_processing_settings = _Cv2Settings(
        max_value=255,
        adaptive_method=cv2.ADAPTIVE_THRESH_MEAN_C,
        threshold_type=cv2.THRESH_BINARY,
        block_size=35,
        adaptive_constant=15,
        kernel_size=15,
        edge_crop_size=50,
        color_format=cv2.COLOR_BGR2GRAY
    )
    filename_in = sys.argv[1]
    filename_output_prefix = f"output/{sys.argv[1]}"

    with open(filename_in, "rb") as in_f:
        file_bytes = in_f.read()
        poppler_pdf = poppler.load_from_data(file_bytes)
        pdf_images = (
            pdf_page_to_pillow_image(file_bytes, page_number + 1)
            for page_number in range(poppler_pdf.pages)
        )
        processed_images = (
            image_to_greyscale(image)
            .bind(invert_image)
            .bind(image_adaptive_threshold)
            .bind(erode_image)
            .map(find_margin_extents)
            .map(get_image_cropper(image))
            (image_processing_settings)
            for image in (
                pillow_image_to_cv2(pdf_image) for pdf_image in pdf_images
            )
        )

        img2pdf_filenames = []

        for count, image in enumerate(processed_images):
            filename = f"{filename_output_prefix}.{count}.jpg"
            cv2.imwrite(filename, image, [cv2.IMWRITE_JPEG_QUALITY, 60])
            img2pdf_filenames.append(filename)

    with open(f"{filename_in}.converted.pdf", "wb") as out_f:
        out_f.write(img2pdf.convert(img2pdf_filenames))
