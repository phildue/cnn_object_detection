from utils.fileaccess.CocoGenerator import CocoGenerator
from utils.image.Imageprocessing import show
from utils.workdir import cd_work

cd_work()
dataDir = 'lib/cocoapi/'
generator = CocoGenerator(dataDir)

for imgs, labels in generator.generate_valid(10, 20):
    for img, label in zip(imgs, labels):
        show(img, labels=label)
