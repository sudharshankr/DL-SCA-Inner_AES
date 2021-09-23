# DL-SCA against Inner Rounds of AES-128

This project focusses on attacking the inner rounds of AES-128, particularly the second and the third rounds. This is implemented using Deep Learning models and compares its performance with that of CPA. The attack has been performed on power traces acquired on a [Pinata development board](https://www.riscure.com/product/pinata-training-target/) based on a 32-bit STM32F4 microcontroller with an ARM-based architecture and running a standard unprotected AES-128 look-up table implementation.

In order to run the DL-SCA attack, follow the below given steps:

1. The traces first need to be extracted from `.trs` files and labelled for profiling, validation and the attack phases. This is done by `read_trs.py`.
2. The profiling is then done by running `sca_dl_train_model.py`. Here we can either choose random models or the CNN<sub>best</sub> model. Currently, this has to be done manually by commenting out the appropriate lines of code in the file.
3. The attack can be done by running `attack.py`. Here too, the appropriate changes would have to be done depending on which model was used during the profiling phase, a random one or CNN<sub>best</sub>.

All the configuration settings is done in `config.ini`. The attack stores the attack results and the plot images for which the local directories need to be made before running the code. 

The directory `cpa_attacks` contains the scripts for CPA. In order to perform the CPA attacks, like done for DL, the traces and their PoIs are first extracted from the given `.trs` file and stored in a `.npz` file. These extracted traces are then used by the attack scripts. As for the attack scripts, as their names suggest, `cpa_round_1.py` performs CPA for round 1 hypothesis, `cpa_round_2.py` for round 2 and `cpa_round_3.py` for round 3.

This project was done as a part of my master thesis at TU Delft. The findings of this project can be found in the eprint archive [here](https://eprint.iacr.org/2021/981.pdf). You are always welcome to contibute by creating pull requests :)