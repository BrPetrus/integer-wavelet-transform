from wavelets.lifting_step import Wavelet, LSStep, LSType


def coiflet1() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [4.6458], 0),
        LSStep(LSType.UPDATE, [-0.2057, -0.1172], 1),
        LSStep(LSType.PREDICT, [-0.6076, 7.4686], -1),
        LSStep(LSType.UPDATE, [0.0729], 2),
    ]


def coiflet2() -> Wavelet:
    return [
        LSStep(LSType.PREDICT, [2.5303], 0),
        LSStep(LSType.UPDATE, [-0.3418], 1),
        LSStep(LSType.PREDICT, [-15.2684, -3.1632], -1),
        LSStep(LSType.UPDATE, [0.0646, -0.0057], 3),
        LSStep(LSType.PREDICT, [-13.5912, 63.9510], -3),
        LSStep(LSType.UPDATE, [0.0019, -5.0873e-04], 5),
        LSStep(LSType.PREDICT, [3.7930], -5)
    ]
