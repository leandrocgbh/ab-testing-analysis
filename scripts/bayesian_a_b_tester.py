import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import logging
import argparse

def plot_groups(p_A_samples, p_B_samples):
    """This function plots both A/B groups posterior distributions in order to compare them.

     Parameters
     ===========
     p_A_Samples: Array-like with samples from A group's posterior distribution
     p_B_Samples: Array-like with samples from B group's posterior distribution
     
    """

    plt.figure(figsize=(8,4))
    plt.hist(p_A_samples, 
            histtype='stepfilled',
            bins=25, 
            alpha=0.6, 
            density=True, 
            color='grey', 
            label='Group A')
    plt.hist(p_B_samples, 
            histtype='stepfilled', 
            bins=25, 
            alpha=0.4, 
            density=True,
            label='Group B')
    plt.title('Density Distribution (A x B)')
    plt.legend()
    plt.show()


def parse_args():
    """This function parses cli (command line interface) arguments passed in runtime. """

    parser = argparse.ArgumentParser(
        description='Bayesian A/B Test Evaluation')

    parser.add_argument('-l',
                        '--loglevel',
                        dest='loglvel',
                        choices=['INFO', 'DEBUG', 'WARNING', 'ERROR'],
                        default='INFO',
                        required=False)

    parser.add_argument('-n_a',
                        '--success_a',
                        dest='n_a',
                        help="Number of success cases from A group",
                        default=None)

    parser.add_argument('-n_b',
                        '--success_b',
                        dest='n_b',
                        help="Number of success cases from B group",
                        default=None)

    parser.add_argument('-N_a',
                        '--total_a',
                        dest='N_a',
                        help="Total cases from A group",
                        default=None)

    parser.add_argument('-N_b',
                        '--total_b',
                        dest='N_b',
                        help="Total cases from B group",
                        default=None)

    parser.add_argument('-p_a',
                        '--priors_a',
                        dest='p_a',
                        help="Prior belief for A group in the format [from, to] e.g [0,1]",
                        default=None,
                        required=False)

    parser.add_argument('-p_b',
                        '--priors_b',
                        dest='p_b',
                        help="Prior belief for B group in the format [from, to] e.g [0,1]",
                        default=None,
                        required=False)
                    
                        
    args = parser.parse_args()

    return args


if __name__ == '__main__':

    # Parsing CLI arguments
    CLI_args = parse_args()

    #Getting Parameters
    n_a = CLI_args.n_a
    n_b = CLI_args.n_b
    N_a = CLI_args.N_a
    N_b = CLI_args.N_b

    if CLI_args.p_a is not None:
        priors_a = CLI_args.p_a
    else:
        priors_a = [0,1]

    if CLI_args.p_b is not None:
        priors_b = CLI_args.p_b
    else:
        priors_b = [0,1]
     
    if CLI_args.loglvel == 'INFO':
        logging.getLogger().setLevel(logging.INFO)
    elif CLI_args.loglvel == 'DEBUG':
        logging.getLogger().setLevel(logging.DEBUG)
    elif CLI_args.loglvel == 'WARNING':
        logging.getLogger().setLevel(logging.WARNING)
    elif CLI_args.loglvel == 'ERROR':
        logging.getLogger().setLevel(logging.ERROR)
    else:
        logging.getLogger().setLevel(logging.INFO)


    ##Bayesian model
    logging.info('Starting Bayesian Model Trainning')

    with pm.Model() as ab_test_model:
        
        #Non informatinve prior estimations for probabilities of success
        p_a = pm.Uniform('p_a', lower=priors_a[0], upper=priors_a[1])
        p_b = pm.Uniform('p_b', lower=priors_b[0], upper=priors_b[1])
        
        #difference between the groups
        delta = pm.Deterministic('delta', p_b - p_a)
        
        #observed data
        obs_a = pm.Binomial('obs_a', p=p_a, n=N_a, observed=n_a)
        obs_b = pm.Binomial('obs_b', p=p_b, n=N_b, observed=n_b)
        
        start = pm.find_MAP()
        step = pm.NUTS() #default
        burned_trace = pm.sample(10000, step=step, start=start, tune=10000)

    logging.info('Finished Bayesian Model Trainning')

    print(az.summary(burned_trace))

    p_A_samples = burned_trace["p_a"]
    p_B_samples = burned_trace["p_b"]
    delta_samples = burned_trace["delta"]

    plot_groups(p_A_samples, p_B_samples)

    az.plot_posterior(data=burned_trace, var_names=['delta'])
    plt.show()