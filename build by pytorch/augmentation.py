import torchvision.transforms.functional as F
import random

class HorizontalFlip(object):
    """Horizontally flip the given PIL Image."""

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be flipped.

        Returns:
            PIL Image: Flipped image.
        """
        b=random.randint(0,1)
        if b==0:
            return img
        else:
            return F.hflip(img)
