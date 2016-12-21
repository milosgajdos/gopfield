package hopfield

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// Neuron is Hopfield network unit
type Neuron struct {
	// state is unit state: +1/-1
	state float64
	// changed signifies neuron state change
	changed bool
}

// Net is Hopfield network
type Net struct {
	// network neurons
	neurons []*Neuron
	// weights are network neurons weights
	weights *mat64.SymDense
	// biases are network unit direct inputs
	biases *mat64.Vector
}

// NewNet creates new Hopfield network and returns it.
// If negative size is supplied NewNet returns error
func NewNet(size int) (*Net, error) {
	// can't have negative number of weights
	if size <= 0 {
		return nil, fmt.Errorf("Invalid network size: %d\n", size)
	}

	// allocate neurons slice and iitialize it
	neurons := make([]*Neuron, size)
	for i, _ := range neurons {
		neurons[i] = new(Neuron)
	}
	// allocate weights and bias matrices
	weights := mat64.NewSymDense(size, nil)
	biases := mat64.NewVector(size, nil)

	return &Net{
		neurons: neurons,
		weights: weights,
		biases:  biases,
	}, nil
}

// Weights returns network weights
func (n Net) Weights() mat64.Symmetric {
	return n.weights
}

// Neurons returns a slice of network neurons
func (n Net) Neurons() []*Neuron {
	return n.neurons
}

// Biases return network biases
func (n Net) Biases() mat64.Matrix {
	return n.biases
}

// Store stores pattern in network. It modifies the network weights matrix using Hebbian learning.
// Store returns error if supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n *Net) Store(pattern []float64) error {
	// pattern can't be nil
	if pattern == nil {
		return fmt.Errorf("Invalid data supplied: %v\n", pattern)
	}
	// pattern length must be the same as number of neurons
	if len(pattern) != len(n.neurons) {
		return fmt.Errorf("Dimension mismatch: %v\n", pattern)
	}
	// remap patterns to +1/-1
	for i := 0; i < len(pattern); i++ {
		if pattern[i] <= 0.0 {
			pattern[i] = -1.0
		} else {
			pattern[i] = 1.0
		}
	}
	// we only traverse higher triangular matrix
	for i := 0; i < len(n.neurons); i++ {
		for j := i + 1; j < len(n.neurons); j++ {
			n.weights.SetSym(i, j, n.weights.At(i, j)+pattern[i]*pattern[j])
		}
	}

	return nil
}

func (n *Net) updateNeuron(idx int, state float64) {
	fmt.Println("Updating neuron", idx, "current state:", n.neurons[idx], "state: ", state)
	if n.neurons[idx].state != state {
		n.neurons[idx].state = state
		n.neurons[idx].changed = true
	}
}

// Restore tried to restor supplied pattern from memory. It starts Hopfield network run which finishes once local minima is found.
// Restore returns a pattern which is the closest to any of the patterns stored in Hopfield network.
// It returns error if the supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n *Net) Restore(pattern []float64, iters int) ([]float64, error) {
	// pattern can't be nil
	if pattern == nil {
		return nil, fmt.Errorf("Invalid data supplied: %v\n", pattern)
	}
	// pattern length must be the same as number of neurons
	if len(pattern) != len(n.neurons) {
		return nil, fmt.Errorf("Dimension mismatch: %v\n", pattern)
	}
	// number of iterations must be a positive integer
	if iters <= 0 {
		return nil, fmt.Errorf("Invalid number of iterations: %d\n", iters)
	}
	// set state of neurons to supplied pattern
	for i := 0; i < len(n.neurons); i++ {
		if pattern[i] <= 0.0 {
			pattern[i] = -1.0
		} else {
			pattern[i] = 1.0
		}
		n.neurons[i].state = pattern[i]
		n.neurons[i].changed = false
	}
	// we will bound the number of iterations to iters
	for i := 0; i < iters; i++ {
		netChanged := false
		// generate pseudorandom sequence
		seq := rand.Perm(len(n.neurons))
		fmt.Println(seq)
		for _, j := range seq {
			sum := 0.0
			for k := 0; k < len(n.neurons); k++ {
				// some all connections to j-th neuron
				sum += n.weights.At(j, k) * pattern[k]
			}
			fmt.Println("Sum:", sum, "neuron", j)
			// update pattern based on result
			switch {
			case sum >= n.biases.At(j, 0):
				pattern[j] = 1.0
			default:
				pattern[j] = -1.0
			}
			// update neurons
			n.updateNeuron(j, pattern[j])
		}
		// check if the network changed
		for _, neuron := range n.neurons {
			// if the network changed continue iterating
			if netChanged = neuron.changed; netChanged {
				break
			}
		}
		// if the network hasn't changed return
		if !netChanged {
			fmt.Println("didnt change")
			return pattern, nil
		}
	}

	return pattern, nil
}

// Energy returns network energy for a given pattern
func (n Net) Energy(pattern []float64) float64 {
	energy := 0.0
	// traverse the network higher triangular weights matrix
	for i := 0; i < len(n.neurons); i++ {
		for j := i + 1; j < len(n.neurons); j++ {
			energy += n.weights.At(i, j) * pattern[i] * pattern[j]
		}
	}
	bias := 0.0
	// calculate the bias additions
	for i := 0; i < len(n.neurons); i++ {
		bias += n.biases.RawVector().Data[i] * pattern[i]
	}

	return -(energy + bias)
}
