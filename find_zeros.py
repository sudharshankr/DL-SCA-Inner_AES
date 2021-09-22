import configparser
import numpy as np

if __name__ == "__main__":
    config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    config.read('config.ini')
    attack_traces_count = config['Traces'].getint('AttackEnd') - config['Traces'].getint('AttackStart')
    leakage_config = config['Leakage']
    training_config = config['Training']
    weights_filename = training_config['WeightsFilename']
    model_parameters_file = training_config['ModelParametersFile']
    traces_filename = config['TRS']['TracesStorageFile']
    byte_attacked = leakage_config.getint('TargetKeyByteIndex')
    leakage = leakage_config.getint('LeakageRound')
    hypothesis = leakage_config.getint('HypothesisRound')
    hyp_type = leakage_config['HypothesisType']
    batch_size = training_config.getint('BatchSize')
    model_id = training_config["ModelId"]
    prefix = training_config["Prefix"]
    results_filename = "../data/attack_results/round_" + str(hypothesis) + "_random_model_results/"+prefix+"/results-model_" + model_id + "-leakage_rnd_" + str(leakage) \
                       + "-hypothesis_rnd_" + str(hypothesis) + "-" + hyp_type + "-" + str(byte_attacked) + ".npz"

    results = np.load(results_filename)
    ranks = results["ranks"]
    zero_pos = np.where(ranks == 0)[0]
    # first_zero = zero_pos[0]
    new_pos = 0
    for i in range(len(zero_pos)-1):
        if zero_pos[i+1] - zero_pos[i] != 1:
            new_pos = zero_pos[i+1]

    with open('zeros_round_3.txt', 'a') as zeros_file:
        zeros_file.write(str(model_id) + ": " + str(zero_pos[0]) + ", " + str(new_pos) + "\n")