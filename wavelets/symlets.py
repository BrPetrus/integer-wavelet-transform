from wavelets.lifting_step import Wavelet, LSStep, LSType


def symlets2() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [-1.7321], 0),
        LSStep(LSType.UPDATE, [0.4330, -0.0670], 1),
        LSStep(LSType.PREDICT, [1], -1),
    ]


def symlets4() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [0.3911], 0),
        LSStep(LSType.UPDATE, [-0.3392, -0.1244], 1),
        LSStep(LSType.PREDICT, [0.1620, -1.4195], 0),
        LSStep(LSType.UPDATE, [0.1460, 0.4313], 0),
        LSStep(LSType.PREDICT, [-1.0493], 1),
    ]


def symlets8() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [0.1603], 0),
        LSStep(LSType.UPDATE, [-0.1563, 0.7103], 1),
        LSStep(LSType.PREDICT, [1.8079, -0.4481], -1),
        LSStep(LSType.UPDATE, [-0.4863, 1.7399], 3),
        LSStep(LSType.PREDICT, [-0.2566, -0.5686], -3),
        LSStep(LSType.UPDATE, [3.7023, -0.8355], 5),
        LSStep(LSType.PREDICT, [-0.3717, 0.5881], -5),
        LSStep(LSType.UPDATE, [0.7492, -2.1581], 7),
        LSStep(LSType.PREDICT, [0.3531], -7),
    ]
