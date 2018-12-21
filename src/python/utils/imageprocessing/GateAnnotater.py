import cv2

from src.python.utils.imageprocessing import Image


class GateAnnotater:
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0),
              (255, 255, 255), (0, 0, 0)]

    def __init__(self, image: Image, ref_pt_init=None):
        self.image = image.array
        self.gate_idx = 0
        if ref_pt_init is None:
            ref_pt_init = [[]]
        self.ref_pt = ref_pt_init
        self.clone = []
        self.clone.append(self.image.copy())

    def record_clicks(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.ref_pt[self.gate_idx]) >= 4:
                self.clone.append(self.image.copy())
                self.ref_pt.append([])
                self.gate_idx += 1

            self.ref_pt[self.gate_idx].append((x, self.image.shape[0] - y))
            cv2.circle(self.image, center=(x, y), radius=1, color=self.colors[self.gate_idx % len(self.colors)],
                       thickness=2)
            cv2.imshow("image", self.image)

    def annotate(self):
        cv2.namedWindow('image')
        cv2.setMouseCallback("image", self.record_clicks)
        # keep looping until the 'q' key is pressed
        while True:
            # display the image and wait for a keypress

            cv2.imshow("image", self.image)
            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
            if key == ord("r"):
                if self.gate_idx >= 1:
                    if len(self.ref_pt[self.gate_idx]) >= 4:
                        self.gate_idx -= 1
                    self.image = self.clone.pop()
                else:
                    self.image = self.clone[0].copy()

                self.ref_pt[self.gate_idx] = []

            # if the 'c' key is pressed, break from the loop
            elif key == ord("c"):
                break
        return self.ref_pt
