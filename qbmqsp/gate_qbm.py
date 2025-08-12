import os
import seaborn as sns
import matplotlib.pyplot as plt
import random
from pennylane import numpy as np
from tqdm import tqdm
#from qbmqsp.qbm import QBM
from .utils import construct_multi_fcqbm_pauli_strings
import seaborn as sns
from .src.utils import import_dataset, split_dataset_labels, split_data
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
import scipy.linalg as spl

import pennylane as qml
from pennylane import numpy as np
from pennylane.pauli.utils import string_to_pauli_word

from .hamiltonian import Hamiltonian
from .qsp_phase_engine import QSPPhaseEngine
from .qevt import QEVT
from .rel_ent import relative_entropy
import matplotlib.colors as mcolors
from functools import partial

import itertools



dev_name='default.qubit'



def generate_pauli_strings_tfim(num_qubits,n_visible,restricted=False,multiple=False):
    """
    Generate Pauli strings for Ising model as a 
    boltzmann machine .
    
    Parameters:
    num_qubits (int): Number of qubits in the quantum Boltzmann machine.
    n_visible (int): Number if visible units.
    restricted(bool) : (False) for unrestricted boltzmann machine
    multiple(bool): (False) Set true for 3-qubit interaction
    
    Returns:
    list: List of Pauli strings representing the Hamiltonian.
    """
    pauli_strings = []

    # Local transverse field terms (X_i)
    for i in range(num_qubits):
        pauli_string = ['I'] * num_qubits
        pauli_string[i] = 'Z'
        pauli_strings.append(''.join(pauli_string))

    # Interaction terms (Z_i Z_j)
    
 
    
    
    
    

    for i, j in itertools.combinations(range(num_qubits), 2):
        if restricted:
            if i<n_visible and j>=n_visible:
                pauli_string = ['I'] * num_qubits
    
                pauli_string[i] = 'Z'
                pauli_string[j] = 'Z'
                pauli_strings.append(''.join(pauli_string))
        else:
            if i<n_visible:
                
                pauli_string = ['I'] * num_qubits
                
                pauli_string[i] = 'Z'
                pauli_string[j] = 'Z'
                   
                pauli_strings.append(''.join(pauli_string)) 

    if multiple:
        
        for i,j,k in itertools.combinations(range(num_qubits), 3):
        
            if restricted:
                if i<n_visible and j>=n_visible:
                    pauli_string = ['I'] * num_qubits
        
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    pauli_strings.append(''.join(pauli_string))
            else:
                
                if i<n_visible and j<n_visible and k<n_visible:
                    
                    
                    pauli_string = ['I'] * num_qubits
                    
                    pauli_string[i] = 'Z'
                    pauli_string[j] = 'Z'
                    pauli_string[k] ='Z'  
                    pauli_strings.append(''.join(pauli_string))         
        
        for i,j,k in itertools.combinations(range(num_qubits), 3):
        
            if i<n_visible and j<n_visible and k>=n_visible:
                
                
                pauli_string = ['I'] * num_qubits
                
                pauli_string[i] = 'Z'
                pauli_string[j] = 'Z'
                pauli_string[k] ='Z'  
                pauli_strings.append(''.join(pauli_string))   
    
    return pauli_strings





def evaluate_qbm(qbm,testing_dataset,cluster,plot=False,quantile=0.95,method='mean'):

    '''
    Evaluates the QBM pn the testing dataset.
    Parameters:
    qbm : QBM instance
    testing_dataset 
    cluster : The number of clusters in the dataset
    quantile
    method : When 'min' is used the energy of a given input vector is taken to be the minimum energy that total configuration achieves.
            In case of 'mean' the average value is used to assign it an energy.

    Returns:
    precision, recall, f1_score
    
    '''
    #training_data=numpy.expand_dims(training_data[:,0],axis=1)
    outliers = qbm.get_binary_outliers(
    dataset=testing_dataset, outlier_index=cluster)

    #outliers=numpy.expand_dims(outliers[:,0],axis=1)
    

    points = qbm.get_binary_cluster_points(
    dataset=testing_dataset, cluster_index=cluster-1)

    #points=numpy.expand_dims(points[:,0],axis=1)
    #print(points)
    predict_points_cluster = np.zeros(len(points), dtype=int)
    predict_points_outliers = np.zeros(len(outliers), dtype=int)
    qbm.calculate_outlier_threshold(quantile, method)
    print("Outlier threshold: ", qbm.outlier_threshold)
    print("Calculate outlier Energy")
    
    testing_data, testing_labels = split_dataset_labels(testing_dataset)
#testing_data=numpy.expand_dims(testing_data[:,0],axis=1)

    outlier_energy = []
    for index, outlier in enumerate(tqdm(outliers), 0):
        outlier = np.reshape(outlier, (qbm.dim_input))
        predict_points_outliers[index], this_outlier_energy = qbm.predict_point_as_outlier(
            outlier,method)
        outlier_energy.append(this_outlier_energy)
    outlier_energy = np.array(outlier_energy)

    o = outlier_energy.reshape((outlier_energy.shape[0]))

    print("Calculate cluster energy")
    cluster_point_energy = []

    for index, point in enumerate(tqdm(points), 0):
        point = np.reshape(point, (qbm.dim_input))
        predict_points_cluster[index], this_cluster_point_energy = qbm.predict_point_as_outlier(
        point,method)
        cluster_point_energy.append(this_cluster_point_energy)
    cluster_point_energy = np.array(cluster_point_energy)

    c = cluster_point_energy.reshape((cluster_point_energy.shape[0]))

    title='test'
#qbmqsp.src.utils.save_output(title="cluster_" + title, object=c)
#QBM.plot_energy_diff([o, c], qbm.outlier_threshold, title + ".pdf")

#QBM.plot_hist(c, o, qbm.outlier_threshold, "qbm_hist" + ".pdf")

########## OUTLIER CLASSIFICATION ##########
    print('Outlier classification: Results...')
    predict_points = np.concatenate(
        (predict_points_cluster, predict_points_outliers))

    

    true_points = np.concatenate(
        (np.zeros_like(cluster_point_energy), np.ones_like(outlier_energy)))

    accuracy, precision, recall = accuracy_score(true_points, predict_points), precision_score(
        true_points, predict_points), recall_score(true_points, predict_points)
    f1 = f1_score(true_points, predict_points)
    tn, fp, fn, tp = confusion_matrix(
        true_points, predict_points, labels=[0, 1]).ravel()

    print(f'Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}, \nNum True Negative: {tn}, Num False Negative: {fn}, Num True Positive: {tp}, Num False Positive: {fp}')

#print(f'Wallclock time: {(end-start):.2f} seconds')
    lab=cluster-1
    print("Outlier threshold: ", qbm.outlier_threshold)
    print("Average clusterpoint energy: ", np.average(cluster_point_energy))
    print("Outlier energy: ", outlier_energy)
    
    if plot==True:
        plt.figure()
        plt.title('Test Dataset')
        sns.scatterplot(x=testing_data[:,0],y=testing_data[:,1])
        sns.scatterplot(x=testing_data[:,0][testing_labels>lab],y=testing_data[:,1][testing_labels>lab], c='r',palette='coolwarm')

        # Actual plotting
        # Display the plot
        fig = plt.figure(0)
        fig.suptitle('Point Energy', fontsize=14, fontweight='bold')

        ax = fig.add_subplot()
        ax.boxplot([o,c], showfliers=False, showmeans=True)
        ax.set_xticklabels(['outlier', 'cluster points'], fontsize=8)

        ax.set_ylabel('Energy')

        plt.axhline(qbm.outlier_threshold)

        plt.plot([], [], '-', linewidth=1, color='orange', label='median')
        plt.plot([], [], '^', linewidth=1, color='green', label='mean')
        plt.legend()

        

    
    
    #plt.title('Predicted Points')
    #sns.scatterplot(x=testing_data[:,0],y=testing_data[:,1], hue=predict_points,palette='coolwarm')
    return precision,recall,f1,[c,o]



class QBM():
    """Quantum Boltzmann machine (QBM) based on quantum signal processing.

    Parameters
    ----------
    β, enc:
        Same as attributes.
    h, θ:
        See qbmqsp.hamiltonian.Hamiltonian
    δ, polydeg:
        See qbmqsp.qsp_phase_engine.QSPPhaseEngine
    hnodes(int): Number of hidden nodes
    epochs(int): Training epochs
    restricted(bool): False by default
    multiple(bool): False by default. Set to True for 3 qubit interaction
    
    Attributes
    ----------
    β : float
        Inverse temperature.
    enc : str in {'general', 'lcu'}
        Unitary block encoding scheme.
    H : qbmqsp.hamiltonian.Hamiltonian
        Constructed from parameters (h, θ).
    qsp : qbmqsp.qsp_phase_engine.QSPPhaseEngine
        Constructed from parameters (δ, polydeg).
    qevt : qbmqsp.qevt.QEVT
        Quantum eigenvalue transform to realize matrix function f(A) = exp(- τ * |A|). Updated after each training epoch.
    observables : qml.operation.Observable
        Observables w.r.t which the QBM is measured to optimize via gradient descent.
    aux_wire, enc_wires, sys_wires, env_wires : list[int]
        Quantum register wires of quantum circuit that prepares and measures the QBM.
    """
    
    
    
    
    
    def __init__(self,data, enc: str, δ: float, polydeg: int, β: float, hnodes,epochs=1,restricted=False,multiple=False) -> None:
        if β < 0:
            raise ValueError("__init__: β must not be negative.")
        
        self.encoded_data, bits_input_vector, num_features = self.binary_encode_data(data, use_folding=True)
        self.dim_input = bits_input_vector * num_features
        self.quantile=0.95
        
        self.n_hidden_nodes=hnodes
        self.dim_input=8
        self.qubits=self.dim_input+self.n_hidden_nodes
        self.h=generate_pauli_strings_tfim(self.qubits,self.dim_input,restricted=restricted,multiple=multiple)   
        nparams = len(self.h)


        θ_init =np.random.rand(nparams)/nparams #np.loadtxt('./weights_7_3_un.txt')
        self.epochs=epochs
        self.β = β
        self.enc = enc
        
        self.H = Hamiltonian(self.h, θ_init)
        self.qsp = QSPPhaseEngine(δ, polydeg)
        self.qevt = self._construct_qevt()
        self.aux_wire, self.enc_wires, self.sys_wires, self.env_wires = self._construct_wires()
        self.observables = self._construct_obervables()
        
       
        
        self.restricted=restricted
        self.multiple=multiple
        
        if multiple:
            if self.restricted:
                self.weights_visible_to_hidden=np.reshape(self.H.θ[self.dim_input+self.n_hidden_nodes:],(self.dim_input,self.n_hidden_nodes))
                self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
                self.biases_visible=self.H.θ[:self.dim_input]
            else:
            
                self.weights_visible_to_visible,self.weights_visible_to_hidden,self.weights_three_body_vvv,self.weights_three_body_vvh=self.get_weights_for_multiple(self.H.θ)
                self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
                self.biases_visible=self.H.θ[:self.dim_input]
        else:    
            
            if self.restricted:
                self.weights_visible_to_hidden=np.reshape(self.H.θ[self.dim_input+self.n_hidden_nodes:],(self.dim_input,self.n_hidden_nodes))
                self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
                self.biases_visible=self.H.θ[:self.dim_input]
            else:
                
                self.weights_visible_to_visible,self.weights_visible_to_hidden=self.get_weights(self.H.θ)
                self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
                self.biases_visible=self.H.θ[:self.dim_input]
        

    def get_binary_cluster_points(self,dataset, cluster_index: int) -> np.ndarray:
        points = np.array([entry[:-1]
                           for entry in dataset if entry[-1] <= cluster_index])

        return self.binary_encode_data(points, use_folding=False)[0]
    
    def get_binary_outliers(self,dataset, outlier_index: int):
        outliers = np.array([entry[:-1]
                            for entry in dataset if entry[-1] >= outlier_index])

        return self.binary_encode_data(outliers, use_folding=False)[0]
  
    def binary_encode_data(self,data, use_folding=False):
        """ Encode a numpy array of form [[numpy.int64 numpy.int64] ...] into a
        list of form [[int, int, int, ...], ...].
        Example: encode [[107  73] [113  90] ...] to
        [[1, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1],[1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0] .
        """

        # find out how many bits we need for each feature
        number_bits = len(np.binary_repr(np.amax(data)))
        number_features = data.shape[1]

        binary_encoded = ((data.reshape(-1, 1) & np.array(2 **
                          np.arange(number_bits-1, -1, -1))) != 0).astype(np.float32)
        if use_folding:
            return binary_encoded.reshape(len(data), number_features*number_bits), number_bits, number_features
        else:
            return binary_encoded.reshape(len(data), number_features, number_bits), number_bits, number_features
    
    
    def n_qubits(self, registers: str | set[str] = None) -> int:
        """Return number of qubits per registers.
        
        Parameters
        ----------
        registers : str | set[str]
            Quantum registers whose number of qubits should be returned.
            Must be an element from or a subset of {'aux', 'enc', 'sys', 'env'}.

        Returns
        -------
        n : int
            Number of qubits used per registers.
        """
        if registers is None:
            registers = {'aux', 'enc', 'sys', 'env'}
        elif type(registers) == str:
            registers = {registers}
        if not registers.issubset({'aux', 'enc', 'sys', 'env'}):
            raise ValueError("n_qubits: registers must be an element from or a subset of %r." % {'aux', 'enc', 'sys', 'env'})
        
        n = 0
        if 'env' in registers:
            n += self.qevt.n_qubits('sys')
        registers.discard('env')
        if len(registers) != 0:
            n += self.qevt.n_qubits(registers)
        return n

    def _generate_qsp_phases(self) -> np.ndarray[float]:
        τ = self.β / (1-self.qsp.δ) * self.H.θ_norm()
        φ = self.qsp.generate(τ)
        return φ

    def _construct_qevt(self) -> QEVT:
        φ = self._generate_qsp_phases()
        h_δ, θ_δ = self.H.preprocessing(self.qsp.δ)
        return QEVT(h_δ, θ_δ, self.enc, φ)
    
    def _construct_wires(self) -> tuple[list[int], list[int], list[int], list[int]]:
        wires = list(range(self.n_qubits()))
        aux_wire = wires[: self.n_qubits('aux')]
        enc_wires = wires[self.n_qubits('aux') : self.n_qubits({'aux', 'enc'})]
        sys_wires = wires[self.n_qubits({'aux', 'enc'}) : self.n_qubits({'aux', 'enc', 'sys'})]
        env_wires = wires[self.n_qubits({'aux', 'enc', 'sys'}) : self.n_qubits({'aux', 'enc', 'sys', 'env'})]
        return aux_wire, enc_wires, sys_wires, env_wires

    def _construct_obervables(self) -> list[qml.operation.Observable]:
        n_aux_enc = self.n_qubits({'aux', 'enc'})
        aux_enc_wires = self.aux_wire + self.enc_wires
        proj0 = qml.Projector( [0] * n_aux_enc, aux_enc_wires)

        new_sys_wires = list(range(self.n_qubits('sys')))
        wire_map = dict(zip(self.sys_wires, new_sys_wires))
        observables = [proj0] + [proj0 @ string_to_pauli_word(self.H.h[i], wire_map) for i in range(self.H.n_params)]
        return observables
    
    def _bell_circuit(self) -> None:
        for i, j in zip(self.sys_wires, self.env_wires):
            qml.Hadamard(i)
            qml.CNOT([i, j])

    def probabilistic(self):
        
        bit_strings=[]
        for i in range(2**(self.n_hidden_nodes+self.dim_input)):
        # Convert the number to its binary representation and pad with leading zeros
            bit_string = bin(i)[2:].zfill(self.n_hidden_nodes+self.dim_input)
             
            bit_list = np.array([int(bit) for bit in bit_string])
            bit_strings.append(bit_list) 
            
      
        sample = random.choices(bit_strings, k=1)

        for i,x in enumerate(sample[0]):
            if x==1:
                qml.PauliX(wires=[self.sys_wires[i]])
    
    def _prepare(self) -> None:
        self._bell_circuit()
        #self.probabilistic()
        self.qevt.circuit(self.aux_wire, self.enc_wires, self.sys_wires)
    
    def _measure(self) -> None:
        #return qml.sample(wires=self.aux_wire+self.enc_wires+self.sys_wires)
        return [qml.expval(self.observables[i]) for i in range(len(self.observables))]
    
    
    def get_sample(self,shots=1):
        dev = qml.device(dev_name,shots=shots, wires=self.n_qubits({'aux','enc','sys'}))
        @qml.qnode(dev)
        
        def quantum_circuit():
            
            self._prepare()
            return qml.sample(wires=self.aux_wire+self.enc_wires+self.sys_wires)
        
        sample=quantum_circuit()
        return sample
    
    def get_average_configuration_from_samples(self, samples: list, input_vector=None):
        ''' Takes samples from Annealer and averages for each neuron and connection
        '''

        # unclamped if input_vector == None
        unclamped = input_vector== None

        # biases (row = sample, column = neuron)
        np_samples = np.vstack(
            tuple([np.array(list(sample.values())) for sample in samples]))
        avgs_biases = np.average(np_samples, axis=0)
        avgs_biases_hidden = avgs_biases[self.dim_input:] if unclamped else avgs_biases
        avgs_biases_visible = avgs_biases[:
                                          self.dim_input] if unclamped else input_vector

        # weights
        avgs_weights_visible_to_hidden = np.zeros(
            self.weights_visible_to_hidden.shape)
        if not self.restricted:
            avgs_weights_visible_to_visible = np.zeros(
                self.weights_visible_to_visible.shape)
        for v in range(self.dim_input):
            # visible to hidden connections
            for h in range(self.n_hidden_nodes):
                x, y = (np_samples[:, v], self.dim_input +
                        h) if unclamped else (input_vector[v], h)
                avgs_weights_visible_to_hidden[v, h] = np.average(
                    x*np_samples[:, y])
            # visible to visible connections
            if not self.restricted:
                for v2 in range(v, self.dim_input):
                    x, y = (np_samples[:, v], np_samples[:, v2]) if unclamped else (
                        input_vector[v], input_vector[v2])
                    avgs_weights_visible_to_visible[v, v2] = np.average(x*y)

        if self.restricted:
            return avgs_biases_hidden, avgs_biases_visible, avgs_weights_visible_to_hidden, None
        else:
            return avgs_biases_hidden, avgs_biases_visible, avgs_weights_visible_to_hidden, avgs_weights_visible_to_visible

    def _compute_expvals(self) -> np.ndarray[float]:
        dev = qml.device(dev_name,wires=self.n_qubits({'aux','enc','sys','env'}))
        #dev = qml.device(dev_name, backend=backend,wires=self.n_qubits(),ibmqx_token=token)
        @qml.qnode(dev)
        
        def quantum_circuit():
            self._prepare()
            return self._measure()
        num_repetitions=1
        avg_measurements=np.zeros(self.H.n_params)
        for repetitions in range(num_repetitions):
            
            measurements = quantum_circuit()
            
            success_probabilty = measurements[0]
            
            
            qbm_expvals = measurements[1:] / success_probabilty
            
            avg_measurements+=qbm_expvals
            
            #print(success_probabilty, "prob")
        
        
        avg_measurements=avg_measurements/float(num_repetitions)
        
       
        return avg_measurements
    
    def _loss_func(self, ρ0: np.ndarray[float], ρ1: np.ndarray[float]) -> float:
        return relative_entropy(ρ0, ρ1, check_state=True).item()
    
    def assemble(self) -> np.ndarray[float]:
        """Assemble QBM."""
        expH = spl.expm(-self.β * self.H.assemble())
        return expH / np.trace(expH)
    
    
    def get_energy(self,input_vector,k=30,method='min'):
        input_vector=[input_vector]
        new_biases=self.biases_hidden+np.matmul(1-2*np.array(input_vector),self.weights_visible_to_hidden).flatten()

        pos_neg=1-2*np.array(input_vector[0])
        if self.multiple:
            
            phases=list(pos_neg[j]*pos_neg[i] for j in range(len(pos_neg)) for i in range(j+1,len(pos_neg)))
            interaction=np.reshape(self.weights_three_body_vvh,(-1,self.n_hidden_nodes))
            
            new_biases=new_biases+np.matmul(phases,interaction)
        
        
        
        # List to store all bit strings
        bit_strings=[]
        p=[]
    # There are 2^n bit strings of length n
        #print(new_biases)
        for i in range(2**self.n_hidden_nodes):
        # Convert the number to its binary representation and pad with leading zeros
            bit_string = bin(i)[2:].zfill(self.n_hidden_nodes)
             
            bit_list = np.array([1-2*int(bit) for bit in bit_string])
            bit_strings.append(bit_list) 
            p.append(np.exp(-self.β*np.dot(bit_list,new_biases)))

        p=np.array(p)
        replacement_value = 10000


        is_inf = np.isinf(p)

# Replace infinite values with the replacement value
        p[is_inf] = replacement_value
        
        probabilities=p/np.sum(p)
        
        sample = random.choices(bit_strings, weights=probabilities, k=k)
        energies= np.dot(sample,new_biases)
        
        if method=='min':
            return np.min(energies)    
        else:
            
             
            avg_energy=np.sum(probabilities*np.dot(np.array(bit_strings),new_biases))
            return avg_energy
            

               

            
     
    
    def free_energy(self,method='min',input_vector=None):
        '''Function to compute the free energy'''

        # calculate hidden term
        
         
        if self.n_hidden_nodes==0: 
            hidden_term=0
        else:
            hidden_term = self.get_energy(method=method,input_vector=input_vector)

       

        # calculate visible_term
        # visible bias
        visible_term = np.dot(
            1-2*np.array(input_vector), self.H.θ[:self.dim_input]) #/beta
        
        pos_neg=1-2*input_vector
        
        if self.restricted==False:
             
             
             vv_interaction=np.matmul(self.weights_visible_to_visible,pos_neg)
             vv_interaction=np.matmul(pos_neg.T,vv_interaction)
             visible_term=visible_term+vv_interaction
        

        if self.multiple:
            
            three_body=list(pos_neg[k]*pos_neg[j]*pos_neg[i] for k in range(len(pos_neg)) for j in range(k+1,len(pos_neg)) for i in range(j+1,len(pos_neg))) 
            
            
            vvv_interaction=np.dot(three_body,self.weights_three_body_vvv)
            visible_term=visible_term+vvv_interaction
        
        return hidden_term + visible_term
    
    def calculate_outlier_threshold(self, quantile=0.95,method='min'):
        
        self.quantile = quantile
        energy_func=partial(self.free_energy,method)
        
        energies = np.apply_along_axis(
            energy_func, axis=1, arr=self.encoded_data)
        
        self.outlier_threshold = np.quantile(energies, self.quantile)
        
        
    
    
    
    
    def get_average_configurations(self,input_vector=None):
        '''
        Function for giving averge configurations of all qubits for the gibbs state of the system.
        Gives configuration over hidden units only, if input vector is clamped at a certain value.
       
    
        Parameters:
        input vector (np.ndarray)
        
    
        Returns:
        list: List of expectation values of hamilatonian terms.
        '''
        
        
        # unclamped values
        if input_vector is None:
            
            qbm_expvals=self._compute_expvals()
            
            return qbm_expvals
        
        # clamped values
        
        
        if self.multiple:
            
        
           
    
            self.weights_visible_to_visible,self.weights_visible_to_hidden,self.weights_three_body_vvv,self.weights_three_body_vvh=self.get_weights_for_multiple(self.H.θ)
            self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
            self.biases_visible=self.H.θ[:self.dim_input]
                
            input_vector=[input_vector]
        
        
            new_biases=self.biases_hidden+np.matmul(1-2*np.array(input_vector),self.weights_visible_to_hidden).flatten()
        #np.matmul(input_vector, self.weights_visible_to_hidden).flatten()
            
            
            phases=[]
            for i, j in itertools.combinations(range(self.dim_input), 2):
        
        
                if i<self.dim_input and j:
                
                  
                    phase = (1-2*input_vector[0][i])*(1-2*input_vector[0][j])
                    
                    
                   
                phases.append(phase) 
            
            phases=np.array(phases)
            
            if self.n_hidden_nodes!=0:
                interaction=np.reshape(self.weights_three_body_vvh,(-1,self.n_hidden_nodes))
                new_biases=new_biases+np.matmul(phases,interaction)
            
                Q_new=new_biases
                exp_vals=-np.tanh(self.β*new_biases)
            else:
                

                exp_vals=[]
            return exp_vals
        
        else:
        
        
        
            if self.restricted:
                self.weights_visible_to_hidden=np.reshape(self.H.θ[self.dim_input+self.n_hidden_nodes:],(self.dim_input,self.n_hidden_nodes))
                self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
                self.biases_visible=self.H.θ[:self.dim_input]
            else:
                
                self.weights_visible_to_visible,self.weights_visible_to_hidden=self.get_weights(self.H.θ)
                self.biases_hidden=self.H.θ[self.dim_input:self.dim_input+self.n_hidden_nodes]
                self.biases_visible=self.H.θ[:self.dim_input]
        
            input_vector=[input_vector]
        
        
            new_biases=self.biases_hidden+np.matmul(1-2*np.array(input_vector),self.weights_visible_to_hidden).flatten()
        #np.matmul(input_vector, self.weights_visible_to_hidden).flatten()
        
            Q_new=new_biases

            exp_vals=-np.tanh(self.β*new_biases)
            return exp_vals
            
        '''
        
        β, δ, θ_norm = self.β, 0.3, np.linalg.norm(Q_new, ord=1)
        τ = β * θ_norm / (1-δ)
        
        φ = self.qsp.generate(τ)
        #del qsp_phase_engine
         
      
        # New energy configuration only for hidden units
        h=generate_pauli_strings_tfim(self.n_hidden_nodes,self.n_hidden_nodes)
        
        h_δ = h + [self.n_hidden_nodes * 'I']
        θ_δ = np.append(Q_new * (1-δ)/(2*θ_norm), (1+δ)/2)
        encoding='general' 
        qevt = QEVT(h_δ, θ_δ, encoding, φ)
        
        
        
        n_aux, n_enc, n_sys = qevt.n_qubits({'aux'}), qevt.n_qubits({'enc'}), qevt.n_qubits({'sys'})
        wires = list(range(n_aux + n_enc + 2*n_sys))
        aux_wire = wires[: n_aux]
        enc_wires = wires[n_aux : n_aux+n_enc]
        sys_wires = wires[n_aux+n_enc : n_aux+n_enc+n_sys]
        env_wires=wires[n_aux+n_enc+n_sys:]
        
        new_sys_wires = list(range(n_sys))
        wire_map = dict(zip(self.sys_wires, new_sys_wires))
        proj0 = qml.Projector( [0] * (n_aux+n_enc), aux_wire+enc_wires)
        observables = [proj0] + [proj0 @ string_to_pauli_word(h[i], wire_map) for i in range(len(h))]
        
        
        dev = qml.device(dev_name, wires=n_aux+n_enc+2*n_sys)
        #dev = qml.device(, backend=backend,wires=n_aux+n_enc+2*n_sys,ibmqx_token=token)
        @qml.qnode(dev)
        
        
        def qevt_circuit():
            for i, j in zip(sys_wires, env_wires):
                qml.Hadamard(i)
                qml.CNOT([i, j])
            qevt.circuit(aux_wire, enc_wires, sys_wires)
            return [qml.expval(observables[i]) for i in range(len(observables))]
       
        measurements = qevt_circuit()
        success_probabilty = measurements[0]
        qbm_expvals = measurements[1:] / success_probabilty    
        
        return qbm_expvals
    
        '''
            
        
        
    def train_for_one_iteration(self, batch, learning_rate):

        errors = 0
        #errors_biases_visible = 0
        #errors_weights_visible_to_hidden = 0
        #if not self.restricted:
          #  errors_weights_visible_to_visible = 0

        for i,input_vector in enumerate(batch):
            
            
            if i==0:
                unclamped_config = self.get_average_configurations() 
               
            
            clamped_config = self.get_average_configurations(input_vector) # has only expectations over hidden units
            
            # avgs_weights_visible_to_visible_clamped only has a value if not restricted
           
            
            # Getting averages for all qubits , avg_visible=input_vector
            
            full_clamped_config=np.zeros_like(unclamped_config)
            
            full_clamped_config[:self.dim_input]=1+(-2)*input_vector   
            full_clamped_config[self.dim_input:self.dim_input+self.n_hidden_nodes]=clamped_config
            
            pos_neg=1-2*input_vector
            
            
            if self.multiple:
                visible=list(pos_neg[j]*pos_neg[i] for j in range(len(pos_neg)) for i in range(j+1,len(pos_neg)))
                    
                hidden=np.kron(pos_neg,clamped_config)
                for i in range(1,self.dim_input+1):
                    for j in range(self.n_hidden_nodes):
                        visible.insert((i-1)*self.n_hidden_nodes+(i)*(self.dim_input-1)-(i-1)+j,hidden[self.n_hidden_nodes*(i-1)+j])
                
                index_multiple=self.dim_input+self.n_hidden_nodes+int((self.dim_input*(self.dim_input-1)/2))+self.dim_input*self.n_hidden_nodes
                
                full_clamped_config[self.dim_input+self.n_hidden_nodes:index_multiple]=np.array(visible)
                
                
                
                vvv_config=list(pos_neg[k]*pos_neg[j]*pos_neg[i] for k in range(len(pos_neg)) for j in range(k+1,len(pos_neg)) for i in range(j+1,len(pos_neg)))
                
                vvh_config=[pos_neg[k]*pos_neg[j]*clamped_config[i] for k in range(len(pos_neg)) for j in range(k+1,len(pos_neg)) for i in range(0,self.n_hidden_nodes)]

                
                full_clamped_config[index_multiple:index_multiple+int(self.dim_input*(self.dim_input-1)*(self.dim_input-2)/6)]=vvv_config
                
                full_clamped_config[index_multiple+int(self.dim_input*(self.dim_input-1)*(self.dim_input-2)/6):]=vvh_config 
            
            else:
                if self.restricted:
                    
                    
                    full_clamped_config[self.dim_input+self.n_hidden_nodes:]=np.kron(pos_neg,clamped_config)
                
                else:
                    
                    
                    
                    visible=list(pos_neg[j]*pos_neg[i] for j in range(len(pos_neg)) for i in range(j+1,len(pos_neg)))
                    hidden=np.kron(pos_neg,clamped_config)
                    for i in range(1,self.dim_input+1):
                        for j in range(self.n_hidden_nodes):
                            visible.insert((i-1)*self.n_hidden_nodes+(i)*(self.dim_input-1)-(i-1)+j,hidden[self.n_hidden_nodes*(i-1)+j])
                    full_clamped_config[self.dim_input+self.n_hidden_nodes:]=np.array(visible)
                
            
            
            
            
            errors += full_clamped_config - unclamped_config
            
            
            
            
            
            
            

        errors /= batch.shape[0]
        
        self.H.θ = self.H.θ - learning_rate * errors
                
        self.qevt = self._construct_qevt()
                
       
        
        
        return np.average(errors[:self.dim_input]**2)
    
    
    
    
    def train_model(self, batch_size=8, learning_rate=0.005,save=False):
        
        data = self.encoded_data
        
        weights=[]
        batch_num = data.shape[0] // batch_size
        diff = data.shape[0] % batch_size
        self.batch_size=batch_size
        if diff:
            
        
            data = data[:-diff]
            last_batch = data[data.shape[0] - diff:]
        
        
        
        batches = np.vsplit(data, batch_num)
        
        if diff:
            batches.append(last_batch)
              
        losses=[]
        
        for epoch in range(1, self.epochs+1):
            print(f'Epoch {epoch}')
            batch_errors = None
            batchnum = 1
            errors_epoch=[]
            for batch in tqdm(batches):
                   
                    errors = self.train_for_one_iteration(batch, learning_rate)
                    
                    if type(batch_errors) is np.ndarray:
                        batch_errors = np.hstack((batch_errors, errors))
                    else:
                        batch_errors = errors
                    #self.save_weights(
                        #f'e{epoch}_b{batchnum}_{self.paramstring}')
                    batchnum += 1
               
                    #self.save_weights(
                     #   f'e{epoch}_b{batchnum}_{self.paramstring}')
                    #raise e
                    errors_epoch.append(errors)
            
            losses.append(errors_epoch)
            weights.append(self.H.θ)
            if save==True:
                try:
                    np.savez(f'./epoch{epoch}_weights_h{self.n_hidden_nodes}_v{self.dim_input}_lr{self.learning_rate}_e{self.epochs}',self.H.θ)
                    np.savez(f'./epoch{epoch}_losses_h{self.n_hidden_nodes}_v{self.dim_input}_lr{self.learning_rate}_e{self.epochs}',errors_epoch)
                except:
                    print('error_saving')
        self.calculate_outlier_threshold(self.quantile)
        
        
        
        return losses, weights 
    
    #self.error_container.add_error(batch_errors)
        #self.error_container.plot("qbm_errors" + self.paramstring)
        #self.save_weights(title="final_weights_qbm" + self.paramstring)
        # make list of values of the error dicts
        
        #self.calculate_outlier_threshold(self.quantile)
       
    def predict_point_as_outlier(self, input,method):
        energy = self.free_energy(method,input)
        if energy >= self.outlier_threshold:
            return 1, energy
        return 0, energy
        
    
    def get_weights_for_multiple(self,Q):
        weights_vh_vv=list(Q[self.dim_input+self.n_hidden_nodes:self.dim_input*self.n_hidden_nodes+int(self.dim_input*(self.dim_input-1)/2)+self.n_hidden_nodes+self.dim_input])
        
        for i in range(1,self.dim_input+1):
            for j in range(i):
                weights_vh_vv.insert((self.dim_input+self.n_hidden_nodes)*(i-1)+j,0)
            
        weights_vh_vv=np.array(weights_vh_vv)
        
        weights_visible_to_visible=weights_vh_vv.reshape(self.dim_input,self.dim_input+self.n_hidden_nodes)[:,0:self.dim_input]
        weights_visible_to_hidden=weights_vh_vv.reshape(self.dim_input,self.dim_input+self.n_hidden_nodes)[:,self.dim_input:]
    
        weights_three_body=Q[self.n_hidden_nodes+self.dim_input+self.dim_input*self.n_hidden_nodes+int(self.dim_input*(self.dim_input-1)/2):] 
        
        weights_three_body_vvv=weights_three_body[:self.dim_input*(self.dim_input-1)*int((self.dim_input-2)/6)]
        weights_three_body_vvh=weights_three_body[self.dim_input*(self.dim_input-1)*int((self.dim_input-2)/6):]
    
        return weights_visible_to_visible,weights_visible_to_hidden,weights_three_body_vvv,weights_three_body_vvh
    
    def get_weights(self,Q):
        weights_vh_vv=list(Q[self.dim_input+self.n_hidden_nodes:])
            
        for i in range(1,self.dim_input+1):
            for j in range(i):
                weights_vh_vv.insert((self.dim_input+self.n_hidden_nodes)*(i-1)+j,0)
            
        weights_vh_vv=np.array(weights_vh_vv)
        
        weights_visible_to_visible=weights_vh_vv.reshape(self.dim_input,self.dim_input+self.n_hidden_nodes)[:,0:self.dim_input]
        weights_visible_to_hidden=weights_vh_vv.reshape(self.dim_input,self.dim_input+self.n_hidden_nodes)[:,self.dim_input:]
    
        return weights_visible_to_visible,weights_visible_to_hidden
    
    def save_model(path,dataset_name):
         path=Path(path/dataset)
         path.mkdir(exist_ok=True)
         np.savez(f'_e{qbm.epochs}_h{qbm.n_hidden_nodes}_v{qbm.dim_input}_b{qbm.batch_size}',qbm.H.θ)
        
     

def plot_pixel_dataset(data,CLUSTER):
    data=data

    x_values = np.arange(0,16,1)
    y_values=np.arange(0,16,1)

    c_values=[]
    for x in x_values:
        row_values=[]
        for y in y_values:
            num=len(data[(data[:,1]==y) & (data[:,0]==x)] )
        
            if num==1:
                if data[(data[:,1]==y) & (data[:,0]==x)][0][2]>=CLUSTER:
            
                    row_values.append(-1)
                else:
                    row_values.append(1)
            else:
                row_values.append(num)
        c_values.append(row_values)

#c_values=np.array(c_values)/np.max(c_values)


# Ensure the x_values array has the same length as the number of columns


    cols = len(x_values)
    rows = len(y_values)
# Number of rows in the grid

# Repeat the x_values array to create a 2D array for the grid

    values=np.array(c_values)
# Normalize the values to [0, 1] for color mapping

    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

# Create a colormap
    cmap = plt.get_cmap('bone_r')
    fig, ax = plt.subplots(figsize=(30,8))

# Plot each cell with a color corresponding to the value
    for i in range(rows):
        for j in range(cols):
            if values[i][j]==0:
                color=(0.0,0.0,1.0,0.3)
            elif values[i][j]==1:
                color=(0.0,0.0,0.0,0.01)
                
            elif values[i][j]==-1:
                color=(1.0,0.0,0.0,1.0)
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color,label='outlier')
            else:
                color = cmap(norm(values[i, j]))
            
            if values[i][j]!=-1:
                rect = plt.Rectangle((j, i), 1, 1, facecolor=color)
            ax.add_patch(rect)

    # Set the limits and aspect ratio
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    
    # Remove the axes for better visualization
    
    ax.set_xticks(np.arange(0.5,16.5,1))
    ax.set_xticklabels(np.arange(0,16,1),rotation=90)
    ax.set_yticks(np.arange(0.5,16.5,1))  # Remove y-axis labels
    ax.set_yticklabels(np.arange(0,16,1),rotation=90)
    # Show the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar=plt.colorbar(sm, ax=ax)
    cbar.ax.set_title('Number of data points')
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    
    
    
    # Add legend with unique labels
    plt.legend(unique_labels.values(), unique_labels.keys())
    
    
    # Display the plot
    plt.show()



    