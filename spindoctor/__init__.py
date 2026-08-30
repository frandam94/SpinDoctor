"""SpinDoctor - basketball spin measurement from video.

End-to-end computer vision pipeline: ball/hand/dot detection (RF-DETR) ->
release-frame detection -> rigid tracking of the dots on the sphere -> spin
axis and rate via Wahba's problem (Davenport's q-method).
"""

__version__ = "1.0.0"
