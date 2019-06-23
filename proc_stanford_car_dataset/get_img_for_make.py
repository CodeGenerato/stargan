import shutil
import os
import ntpath
from PIL import Image
import math

def print_makes(models):
    makes = set()
    for cl in models:
        brand = cl.split(" ")[0]
        makes.add(brand)

    ordered_makes = list(makes)
    ordered_makes.sort()
    print(ordered_makes)


def get_BB_for_img(name, annos):
    for it in annos:
        base, file_name = ntpath.split(it[0])
        if(file_name == name):
            return ((int)(it[1]), (int)(it[2]), (int)(it[3]), (int)(it[4]))


def get_files_per_make_dict(annos, models):
    obsv = {}
    for it in annos:
        class_ind = (int)(it[5])
        model = models[class_ind-1]
        make_name = model.split(" ")[0]

        base, file_name = ntpath.split(it[0])

        if make_name in obsv:
            v = [file_name]
            v.extend(obsv[make_name])
            obsv[make_name] = v
        else:
            obsv[make_name] = [file_name]

    return obsv


def file_to_list(path):
    annos = []
    with open(path) as f:
        f.readline()
        for line in f:
            annos.append(line.split(","))
    return annos

def resize_aspec(img, bb, tgt_resol, fit_image=True):
    x1, y1, x2, y2 = bb
    width = x2-x1
    height = y2-y1

    if width > height:
        ydiff = (width-height)/2
        y1 = y1-(int)(ydiff)
        y2 = y2+math.ceil(ydiff)
    elif height > width:
        xdiff = (height-width)/2
        x1 = x1-(int)(xdiff)
        x2 = x2+math.ceil(xdiff)

    if fit_image:
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img.width, x2)
        y2 = min(img.height, y2)

    cropped = img.crop((x1, y1, x2, y2))
    cropped.load()
    cropped = cropped.resize((tgt_resol, tgt_resol), Image.ANTIALIAS)

    return cropped


def main():
    ann = "./annotations.csv"
    cl = "./class_names.csv"
    image_dir = "/home/nico/Dokumente/Entwicklung/Uni/HLCV_Data/standfor_cars/car_ims"
    tgt_base_dir = "/home/nico/Dokumente/Entwicklung/Uni/HLCV_Data/standfor_cars"
    tgt_resolution = 256

    annos = file_to_list(ann)
    models = file_to_list(cl)[0]

    # print available makes
    print_makes(models)

    mf_dict = get_files_per_make_dict(annos, models)
    items = mf_dict.items()

    # analyze
    # items_ex = list(map(lambda t: (t[0], len(t[1]), t[1]), items))
    # count_sort = sorted(items_ex, key=lambda tup: tup[1], reverse=True)
    # name_sort = sorted(items_ex, key=lambda tup: tup[0])

    make_filter = ["BMW", "Audi", "Mercedes-Benz", "Rolls-Royce"]
    obsv_filtered = list(filter(lambda t: t[0] in make_filter, items))
    print(obsv_filtered)

    for name, files in obsv_filtered:
        tgt_dir = tgt_base_dir+"/"+name
        if os.path.exists(tgt_dir):
            shutil.rmtree(tgt_dir)
            os.mkdir(tgt_dir)
        else:
            os.mkdir(tgt_dir)
        for f in files:
            img_path = image_dir+"/"+f

            bb = get_BB_for_img(f, annos)

            imageObject = Image.open(img_path)
            img = resize_aspec(imageObject, bb, tgt_resolution, True)
            img.save(tgt_dir+"/" + f, "JPEG")


if __name__ == "__main__":
    main()