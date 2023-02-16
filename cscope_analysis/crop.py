import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RectangleSelector


class cScopeCropGUI(object):
    def __init__(self, vcap):

        self.vcap = vcap
        self.crop_coords = None

    def _create_plot(self):

        plt.ion()
        plt.title("Crop cScope Video")

        self.plt_image = plt.imshow(self._get_image())

        self.next_button = Button(plt.axes([0.3, 0.025, 0.15, 0.075]), "Next Image")
        self.next_button.on_clicked(self._update_image)

        self.set_button = Button(plt.axes([0.55, 0.025, 0.15, 0.075]), "Set")
        self.set_button.on_clicked(self._set_crop)

        # self.crop_rectangle = RectangleSelector(
        #     self.plt_image.axes,
        #     self._draw_rectangle,
        #     drawtype="box",
        #     rectprops={"edgecolor": "red", "fill": False},
        # )

    def _get_image(self):

        ret, frame = self.vcap.read()
        if ret:
            return frame
        else:
            raise Exception("Crop GUI: OpenCV VideoCapture did not return an image!")

    def _update_image(self, event=None):

        self.plt_image.set_data(self._get_image())

    def _set_crop(self, event=None):

        xmin, xmax = self.plt_image.axes.get_xlim()
        ymax, ymin = self.plt_image.axes.get_ylim()

        self.crop_coords = (
            int(max(0, xmin)),
            int(min(xmax, 752)),
            int(max(ymin, 0)),
            int(min(ymax, 480)),
        )

    def crop(self):

        plt.ion()
        self._create_plot()
        plt.show(block=True)
        plt.ioff()
        return self.crop_coords
