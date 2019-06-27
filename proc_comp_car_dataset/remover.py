import os
import re
import sys


def main():
    if len(sys.argv) < 2:
        print('usage: python remover.py <path/to/dataset>')
        return
    image_dir = sys.argv[1]
    # image_dir="/media/shadowwalkers/DATA/comp-cars/dataset/data/waste"
    mark="X_"

    images=os.listdir(image_dir)

    bmw = re.compile('81-1(09|10|12|13|15|21)-.*')
    merc = re.compile('77-1(57|62|63|74|78|79|83|85)-.*')
    #merc2 = re.compile('77-124-2011-!(5d50|a343).*')
    merc2 = re.compile('77-124-2011-.*') #removes more than neccessary
    audi = re.compile('78-9-2004-*|78-48-2011*|78-53-*')
    volvo = re.compile('111-1715-*')
    extras = re.compile('78-2-2010-fef92bcd5709e2.jpg|78-23-2010-cc7774a8d47a16.jpg|95-910-2008-109684f2440252.jpg|78-50-2009-07cb70c06defb2.jpg|111-1701-2008-60a6530*|111-1701-2009-fc2fde698*|111-1701-2010-aa917b02f9bc8a.jpg|111-1707-2010-331edbe63321b1.jpg|111-1707-2010-ab6d085c24cd82.jpg|111-1708-2014-c19d62b4d0de8b.jpg|111-1714-2012-b4f1744a1051bb.jpg|78-12-2014-f61541f1c7fc5a.jpg|78-13-2014-1d40c0c0142afc.jpg|78-2-2010-fef92bcd5709e2.jpg78-23-2010-cc7774a8d47a16.jpg|78-24-2010-970467a3f4724a.jpg|78-3-2010-d6f15eaecf2118.jpg|78-33-2010-3181f41d1b08ac.jpg|78-33-2014-6770d43d7c9a07.jpg|78-33-2014-6ee6b7ff9a8035.jpg|78-34-2010-a7d06bdc878af0.jpg|78-47-2014-cb09665db1e996.jpg|78-50-2009-13c989ec1c0c26.jpg|78-50-2009-b3fe5d5f970e96.jpg|78-50-2010-1db77e26eec58b.jpg|78-50-2010-240164e58d3e88.jpg|78-50-2010-6d10babd60dbca.jpg|78-50-2010-96c8361149970f.jpg|78-50-2010-b71da5c629ead9.jpg|95-911-2010-0b8e6ff0b181dd.jpg|95-924-2014-dfbbf283851c57.jpg')
    patterns=[]
    patterns.append(bmw)
    patterns.append(merc)
    patterns.append(merc2)
    patterns.append(audi)
    patterns.append(volvo)
    patterns.append(extras)

    for img in images:
        for p in patterns:
            if p.match(img):
                os.rename(image_dir+"/"+img,image_dir+"/"+mark+img)
                break


if __name__ == '__main__':
    main()

