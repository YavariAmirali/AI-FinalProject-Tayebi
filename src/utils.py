def crop_center_10_percent(img):
    """
    Centralized function for the critical 10% crop.
    """
    h, w = img.shape[:2]
    crop_fraction = 0.10

    start_y = int(h * crop_fraction)
    end_y = int(h * (1 - crop_fraction))
    start_x = int(w * crop_fraction)
    end_x = int(w * (1 - crop_fraction))

    return img[start_y:end_y, start_x:end_x]