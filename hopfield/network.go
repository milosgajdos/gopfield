package hopfield

import (
	"fmt"
	"math/rand"

	"github.com/gonum/matrix/mat64"
)

// Neuron is Hopfield network unit
type Neuron struct {
	// state is neuron's state: +1/-1
	state float64
}

// ChangeState changes the state of Neuron and returns true if the state has changed. Otherwise it returns false.
func (n *Neuron) ChangeState(state float64) bool {
	if state*n.state < 0.0 {
		n.state = -n.state
		return true
	}

	return false
}

// Net is Hopfield network
type Net struct {
	// network neurons
	neurons []*Neuron
	// weights are network neurons weights
	weights *mat64.SymDense
	// bias are network unit direct inputs
	bias *mat64.Vector
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
	bias := mat64.NewVector(size, nil)

	return &Net{
		neurons: neurons,
		weights: weights,
		bias:    bias,
	}, nil
}

// Neurons returns a slice of network neurons
func (n Net) Neurons() []*Neuron {
	return n.neurons
}

// Weights returns network weights
func (n Net) Weights() mat64.Symmetric {
	return n.weights
}

// Bias return network bias as a vector
func (n Net) Bias() mat64.Matrix {
	return n.bias
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
	// we only traverse higher triangular matrix because we are using Symmetric matrix
	for i := 0; i < len(n.neurons); i++ {
		for j := i + 1; j < len(n.neurons); j++ {
			n.weights.SetSym(i, j, n.weights.At(i, j)+pattern[i]*pattern[j])
		}
	}

	return nil
}

// Restore tries to restore supplied pattern. It runs Hopfield network until either a local minima (equilibrium) is found or maximum
// number of iterations has been reached. Equilibrium is measured by eqiters parameter such that if the network state
// hasn't changed in eqiters number of iteraions we assume the network has reached its energy equilibrium.
// Restore returns a pattern which is the closest to any of the patterns stored in the network.
// It returns error if the supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n *Net) Restore(pattern []float64, maxiters, eqiters int) ([]float64, error) {
	// pattern can't be nil
	if pattern == nil {
		return nil, fmt.Errorf("Invalid data supplied: %v\n", pattern)
	}
	// pattern length must be the same as number of neurons
	if len(pattern) != len(n.neurons) {
		return nil, fmt.Errorf("Dimension mismatch: %v\n", pattern)
	}
	// number of max iterations must be a positive integer
	if maxiters <= 0 {
		return nil, fmt.Errorf("Invalid number of max iterations: %d\n", maxiters)
	}
	// number of equlibrium iterations must be a positive integer
	if eqiters <= 0 {
		return nil, fmt.Errorf("Invalid number of equilibrium iterations: %d\n", eqiters)
	}
	// set neurons states to the pattern
	for i, neuron := range n.neurons {
		neuron.state = pattern[i]
	}
	// we will bound the number of iterations to eqiters and maxiters
	eqiter, maxiter := 0, 0
	for maxiter < maxiters {
		// generate pseudorandom sequence
		seq := rand.Perm(len(n.neurons))
		//fmt.Println(seq)
		for _, i := range seq {
			sum := 0.0
			for j := 0; j < len(n.neurons); j++ {
				// some all connections to j-th neuron
				sum += n.weights.At(i, j) * pattern[j]
			}
			// update pattern based on result
			switch {
			case sum >= n.bias.At(i, 0):
				pattern[i] = 1.0
			default:
				pattern[i] = -1.0
			}
			// update neuron if its state has changed
			switch n.neurons[i].ChangeState(pattern[i]) {
			case true:
				// if the network state changed, reset the counter
				eqiter = 0
			default:
				// if the network state hasnt changed, we are around equlibrium
				eqiter++
			}
			// if we are around equlibrium, exit
			if eqiter == eqiters {
				return pattern, nil
			}
		}
		maxiter++
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
		bias += n.bias.RawVector().Data[i] * pattern[i]
	}

	return -(energy + bias)
}
