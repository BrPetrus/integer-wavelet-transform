import numpy as np

from wavelets.db import db8_wavelet, db4_wavelet, db2_wavelet

def main():
    array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    print("Daubechies 2 wavelet integer transform")
    print("======================================")
    lsw = db2_wavelet()
    approx, diff = array[::2], array[1::2]
    for step in lsw:
        approx, diff = step.evaluate(approx, diff)
        print(f"approx = {approx}, diff = {diff}")
    print(approx)
    print(diff)

    print("Inverse")
    lsw = db2_wavelet()
    cp_approx, cp_diff = approx, diff
    for step in reversed(lsw):
        approx, diff = step.evaluate(approx, diff, inverse=True)
        print(f"approx = {approx}, diff = {diff}")
    result = np.zeros(array.shape)
    result[::2] = approx
    result[1::2] = diff
    print(result)

    print("\n\n\n\n")

    # print(db4_ls_single(array))
    # print("--- .... ----")

    print("Daubechies 4 wavelet integer transform")
    print("======================================")

    approx, diff = array[::2], array[1::2]
    db4 = db4_wavelet()
    for step in db4:
        approx, diff = step.evaluate(approx, diff)
        print(approx, diff)
    print(approx)
    print(diff)

    print("Inverse")
    print("=====================================")
    db4 = db4_wavelet()
    for step in reversed(db4):
        approx, diff = step.evaluate(approx, diff, inverse=True)
    result = np.zeros(array.shape)
    result[::2] = approx
    result[1::2] = diff
    print(result)

    print("\n\n\n\n")

    print("Daubechies 8 wavelet integer transform")
    print("======================================")

    approx, diff = array[::2], array[1::2]
    db8 = db8_wavelet()
    for step in db8:
        approx, diff = step.evaluate(approx, diff)
        print(approx, diff)
    print(approx)
    print(diff)

    print("Inverse")
    print("=====================================")
    for step in reversed(db8):
        approx, diff = step.evaluate(approx, diff, inverse=True)
    result = np.zeros(array.shape)
    result[::2] = approx
    result[1::2] = diff
    print(result)

if __name__ == "__main__":
    main()