qsip-practical-anonymity
=============
Implementation of the set of protocols proposed by Unnikrishnan *et al.*<sup>1</sup> to guarantee the anonymity of two parties, the sender and the receiver, when they wish to transmit a message through the network.

## Introduction
We implemented and simulated the anonymity protocol proposed by Unnikrushnan *et al.*<sup>1</sup>. We also simulated a scenario in which the gates are noisy in order to evaluate the effects of the *depolarining channel* on the gates.

To run the protocols, you need SimulaQron and Qiskit. In this project, we used the following versions:
Qiskit: 0.10.5
SimulaQron: 3.0.4


[1] [A. Unnikrishnan, I. J. MacFarlane, R. Yi, E. Diamanti, D. Markham, I. Kerenidis: "Anonymity for practical quantum networks". Physical Review Letters, volume 122, 24, 2018](https://arxiv.org/abs/1811.04729)

## Simulation of the anonymity protocol
First of all, in order to simulate the anonymity network, we need to create a file where we specify the small rotations to apply to qubits in order to reach a certain fidelity. In fact SimulaQron does not allow to simulate complex noisy states.
So we have to run the following command:

```
python fidelity.py nodes fidelity
```

where *fidelity* is the fidelity that we want to have with the ghz state, *nodes* is the number of the agents in the network.

Now we are ready to simulate the anonymity system running the following command:

```
python createNetwork.py args
```

where *args* are the options we can set. A list of the available options can be obtained with the following commands: `$ python createNetwork.py -h` or `$ python createNetwork.py --help`

If we want, for example, simulate a network with 3 agents where they share a ghz state with fidelity=0.9 and one adversary, the command we have to run is:

```
python createNetwork.py -a 1 -n 3 -f 0.9
```

## Simulation of the *Verification protocol* with noisy gates
For the simulation of the noisy gates, we used three schemes. 
The first two schemes simulate noisy *I* gates at the beginning or at the end of the circuit that generates the ghz state. We can simulate this case running the following command:

```
python verification_simple_noise.py e_o nodes fidelity
```

where:
- 'nodes' is the number of the agents in the network
- 'fidelity' is the fidelity of the ghz state shared
- 'e_o' could be the letter 'e' or the letter 'o'. If we input 'e', we'll run the *Verification* protocol using only even multiples of &#960;, otherwise with 'o' will do the same but with odd multiples.

The third scheme will instead simulate a scenario where all the gates are noisy. This is possible using a command very similar to the previous one:

```
python verification_noisy_gates.py e_o nodes fidelity
```

