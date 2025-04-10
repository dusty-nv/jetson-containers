#!/usr/bin/env python3
print('testing pixsfm...')

from pixsfm.refine_hloc import PixSfM
refiner = PixSfM(conf={"dense_features": {"use_cache": True}})

print('pixsfm OK\n')