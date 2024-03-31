# import argparse
#
# import matplotlib.pyplot as plt
# import numpy as np
# import skimage.io as skio
# from numpy.typing import NDArray
#
# SAVE_IMG = True
#

from wavelets.lifting_step import Wavelet, LSStep, LSType, LSBoundaryCondition


def haar_wavelet() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-1], 0, LSBoundaryCondition.ZERO_PADDING),
        LSStep(LSType.UPDATE, [0.5], 0, LSBoundaryCondition.ZERO_PADDING)
    ]


# def _get_quadrant_bounds(r0, c0, shape):
#     """Get the bounds of particular quadrant."""
#     quadrant_bounds = [
#         (r0, r0 + shape[0], c0, c0 + shape[1]),  # Top left
#         (r0, r0 + shape[0], c0 + shape[1], c0 + 2 * shape[1]),  # Top right
#         (r0 + shape[0], r0 + 2 * shape[0], c0, c0 + shape[1]),  # Bottom left
#         (r0 + shape[0], r0 + 2 * shape[0], c0 + shape[1], c0 + 2 * shape[1])
#         # Bottom right
#     ]
#     return quadrant_bounds
#
#
# def lifting_scheme(img: NDArray[np.int32]) -> tuple[
#         NDArray[np.int32], int, list[tuple[int, int, int, int]]]:
#     """
#     Run the integral discrete wavelet Haar transform on an image.
#
#     Parameters
#     ----------
#     img: numpy array of type int32
#         The image to decompose.
#
#     Returns
#     -------
#     out: Tuple
#         The tuple contains the decomposed image, level of decomposition, a list
#         describing which indices were decomposed, and finally a list containing
#         top left corners of images, created during the decomposition (since
#         we are not always decomposing the approximation coefficients).
#
#     """
#     J = np.floor(np.log2(np.min(img.shape))).astype(int)
#     decomposition = img.copy()
#
#     shape = np.array(img.shape)
#     r0, c0 = 0, 0
#     quadrants = []  # Store which quadrants where decomposed
#     top_left_indices = []
#     for i in range(J):
#         if i != 0:
#             # Choose which quadrant to decompose based on the variance
#             var_bounds = _get_quadrant_bounds(r0, c0, shape)
#             variances = []
#             for bound in var_bounds:
#                 variances.append(np.var(
#                     decomposition[bound[0]:bound[1], bound[2]:bound[3]]))
#             var_idx = np.argmax(
#                 variances)  # Find the one with highest variance
#             bounds = var_bounds[var_idx]
#             quadrants.append(var_idx)
#         else:
#             bounds = (0, shape[0], 0, shape[1])
#
#         # Calculate
#         decomposition[bounds[0]:bounds[1], bounds[2]:bounds[3]] = ls_2D_single(
#             decomposition[bounds[0]:bounds[1], bounds[2]:bounds[3]]
#         )
#
#         shape //= 2
#         if i != 0:
#             top_left_indices.append((r0, c0))
#             r0, c0 = bounds[0], bounds[2]
#
#     assert i == len(quadrants)
#     return decomposition, i, quadrants, top_left_indices
#
#
# def lifting_scheme_inverse(img: NDArray[np.int32], level: int,
#                            quadrants: list[int], decomposition_indices) -> \
#         NDArray[np.int32]:
#     quadrants.reverse()
#     reconstruction = img.copy()
#     shape = np.array(img.shape) // 2 ** level
#     for i in range(level + 1):
#         r0, c0 = decomposition_indices[len(decomposition_indices) - 1 - i]
#
#         # Choose which quadrant to reconstruct
#         if i == level:
#             bounds = 0, shape[0], 0, shape[1]
#         else:
#             quadrant_bounds = _get_quadrant_bounds(r0, c0, shape)
#             bounds = quadrant_bounds[quadrants[i]]
#
#         reconstruction[bounds[0]:bounds[1], bounds[2]:bounds[3]] = \
#             ls_2d_inverse(
#                 reconstruction[bounds[0]:bounds[1], bounds[2]:bounds[3]])
#
#         shape *= 2
#     return reconstruction
#
#
# def ls_2D_single(data: NDArray[np.int32]) -> NDArray[np.int32]:
#     """Single level of decomposition in 2D."""
#
#     def run_ls(array: NDArray[np.int32]) -> NDArray[np.int32]:
#         """Single level of decomposition in 1D."""
#         rows, cols = array.shape
#
#         # First run the lifting scheme on each row
#         f_even = array[::, ::2]
#         f_odd = array[::, 1::2]
#
#         # Predict step
#         f_odd = f_odd - f_even.astype(np.int32)
#
#         # Update
#         f_even = f_even + (f_odd / 2).astype(np.int32)
#
#         array[::, :cols // 2] = f_even
#         array[::, cols // 2:] = f_odd
#         return array
#
#     # Run along the rows
#     data = run_ls(data.T).T
#
#     # The along the columns
#     return run_ls(data)
#
#
# def ls_2d_inverse(data: NDArray[np.int32]) -> NDArray[np.int32]:
#     """Single level inverse decomposition in 2D."""
#
#     def ls_single_inverse(array: NDArray) -> NDArray:
#         """Single level inverse decomposition in 1D."""
#         rows, cols = array.shape
#         approx, diff = array[::, :cols // 2], array[::, cols // 2:]
#
#         # Normalise, Update, Predict, Merge
#         approx = approx - (diff / 2).astype(np.int32)
#         diff = diff + approx.astype(np.int32)
#         array[::, ::2] = approx
#         array[::, 1::2] = diff
#         return array
#
#     # First by columns
#     result = ls_single_inverse(data)
#
#     # Now by rows
#     return ls_single_inverse(result.T).T
#
#
# def main(img: NDArray) -> None:
#     # Ensure it is a grayscale image
#     if len(img.shape) == 3:
#         img = np.mean(img, axis=2)
#
#     assert len(img.shape) == 2
#
#     if np.max(img) > 2 ** 32 - 1:
#         print("[Warning] Loss of precision: The image contain intensities,"
#               "which do not fit into the int32 type!")
#     # Convert to int32
#     img = img.astype(np.int32)
#
#     decomposition, level, quadrants, quadrants_indices = \
#         (lifting_scheme(img.copy()))
#     print(f"Decomposed into {level} levels!")
#     reconstruction = lifting_scheme_inverse(decomposition.copy(), level,
#                                             quadrants, quadrants_indices)
#
#     # Visualise the image, decomposition, and the reconstructed image.
#     fig, axes = plt.subplots(ncols=3)
#     ax_sig, ax_filt, ax_rec = axes
#     ax_sig.imshow(img, cmap="gray")
#     ax_sig.set_title("Original image")
#     ax_filt.imshow(np.log(np.abs(decomposition) + 1), cmap="gray")
#     ax_filt.set_title("Haar decomposition")
#     ax_rec.imshow(reconstruction, cmap="gray")
#     ax_rec.set_title("Reconstruction")
#     for ax in axes:
#         ax.axis('off')
#     fig.tight_layout()
#     fig.savefig("lifting-scheme.png", dpi=300)
#     plt.show()
#
#     # Visualise the difference between original and reconstructed image.
#     fig, axes = plt.subplots(ncols=2)
#     ax_img, ax_diff = axes.flatten()
#     ax_img.imshow(img, cmap="gray")
#     ax_img.set_title("Original image")
#     ax_diff.imshow(img - reconstruction, cmap="gray")
#     ax_diff.set_title("Difference")
#     for ax in axes:
#         ax.axis('off')
#     fig.tight_layout()
#     plt.show()
#
#     diff = img - reconstruction
#     print(f"Mean of differences = {np.mean(diff)}\n"
#           f"Std of differences = {np.std(diff)}")
#
#     if SAVE_IMG:
#         skio.imsave("Reconstruction.jpg", reconstruction)
#         diff = (diff - diff.min()) / np.ptp(diff) * 255
#         diff = diff.astype(np.uint8)
#         skio.imsave("Diff.jpg", diff)
#         skio.imsave("Grayscale.jpg", img)
#         decomposition = np.log(np.abs(decomposition) + 1)
#         decomposition = (decomposition - decomposition.min()) \
#             / np.ptp(decomposition)
#         decomposition *= 256
#         decomposition = decomposition.astype(np.uint8)
#         skio.imsave("Decomposition.jpg", decomposition)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(
#         description='Visualise Haar decomposition.')
#     parser.add_argument('-f', '--file', help='Path to the file.',
#                         required=True)
#     args = vars(parser.parse_args())
#     try:
#         img = skio.imread(args['file'])
#         main(img)
#     except FileNotFoundError as err:
#         print(f"[Error] File with path {args['file']} was not found!")
