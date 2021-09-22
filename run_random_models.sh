#!/bin/bash

filename='config.ini'

for count in {11..12}
do
  sed -i '' "s/ModelId = $((count-1))/ModelId = $count/" $filename
  python3 sca_dl_train_model.py
  python3 attack.py
  python3 find_zeros.py
done

