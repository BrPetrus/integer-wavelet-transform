from PIL import Image
import json
import numpy as np
import tifffile

from wavelets.misc import Decomposition


def save_decomposed_img(decomposition: Decomposition, path: str) -> None:
    coefficients, config = decomposition
    config = config.copy()
    coefficients = coefficients.copy()  # TODO: replace with non-inplace operations

    full_r, full_c = coefficients[0].shape  # Final shape
    full_r *= 2
    full_c *= 2

    with tifffile.TiffWriter(path) as tif_writer:
        # Write the first IFD for the approximation coefficient
        approx = coefficients.pop()
        metadata = {
            'original_row_count': approx.shape[0],
            'original_column_count': approx.shape[1],
        }
        tif_writer.write(
            data=approx,
            metadata=metadata,
        )

        # Now create an IFD for the detail coefficients
        for orig_rows, orig_cols in reversed(config):
            diff_h = coefficients.pop()
            diff_diag = coefficients.pop()
            diff_vert = coefficients.pop()

            img = np.zeros((diff_h.shape[0], diff_h.shape[1], 3), dtype=diff_h.dtype)
            img[:, :, 0] = diff_h
            img[:, :, 1] = diff_diag
            img[:, :, 2] = diff_vert

            metadata = {
                'original_row_count': orig_rows,
                'original_column_count': orig_cols,
            }
            tif_writer.write(
                data=img,
                metadata=metadata,
                photometric=tifffile.tifffile.PHOTOMETRIC.RGB  # TODO: change this
            )


def read_decomposed_img(path: str) -> Decomposition:
    with tifffile.TiffFile(path) as tif:
        coefficients = []
        config = []

        # Extract aproximation coefficient
        approx = tif.pages[0].asarray()

        # Extract metadata
        metadata = json.loads(tif.pages[0].description)

        if approx.shape[0] != metadata['original_row_count'] \
                or approx.shape[1] != metadata['original_column_count']:
            raise RuntimeError(f"Image shape mismatch for the approximation coefficients. Metadata: {metadata}. "
                               f"Got {approx.shape}")
        datatype = approx.dtype
        coefficients.append(approx)

        # Extract detail coefficients
        for i, page in enumerate(tif.pages[1:]):
            data = page.asarray()
            if datatype != data.dtype:
                raise RuntimeError(f"Image datatype mismatch. Page #{i} has {datatype} but expected {data.dtype}.")
            metadata = json.loads(page.description)
            orig_rows, orig_cols = metadata['original_row_count'], metadata['original_column_count']
            orig_shape = (orig_rows, orig_cols)
            config.append(orig_shape)

            diff_h = data[:, :, 2]
            diff_diag = data[:, :, 1]
            diff_vert = data[:, :, 0]
            coefficients.append(diff_vert)
            coefficients.append(diff_diag)
            coefficients.append(diff_h)
    return coefficients[::-1], config[::-1]


if __name__ == "__main__":
    from wavelets.haar import haar_wavelet
    from wavelets import wt_2d, wt_2d_inv

    with Image.open('../tests/resources/maly_rozsutec_2023_grayscale.jpg') as img:
        img = np.array(img).astype(np.int32)

        decomposition = wt_2d(img, haar_wavelet(), 3)

        save_decomposed_img(decomposition, "decomposed_img.tif")

        decomposition_found = read_decomposed_img("./decomposed_img.tif")

        inv_img = wt_2d_inv(decomposition_found, haar_wavelet())

        assert np.all(np.isclose(img, inv_img))
        assert np.all(img == inv_img)
