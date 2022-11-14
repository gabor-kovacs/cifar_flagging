import numpy as np

unlabeled_imgs = np.load("data/unlabeled_imgs.npy")
unlabeled_imgs_part1, unlabeled_imgs_part2 = np.split(unlabeled_imgs, 2)
np.save('data/unlabeled_imgs_part1.npy', unlabeled_imgs_part1)
np.save('data/unlabeled_imgs_part2.npy', unlabeled_imgs_part2)

unlabeled_imgs_part1_loaded = np.load("data/unlabeled_imgs_part1.npy")
unlabeled_imgs_part2_loaded = np.load("data/unlabeled_imgs_part2.npy")
assembled = np.append(unlabeled_imgs_part1_loaded, unlabeled_imgs_part2_loaded, axis=0)

assert assembled.all() == unlabeled_imgs.all()

