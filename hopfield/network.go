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
	for i := range neurons {
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

// Store stores supplied patterns in network. It modifies the network weights based on specified learning method.
// If an unsupported learning method is provided, store defaults to Hebbian learning.
// Store returns error if ps is nil or if any of the data patterns do not have the same dimension as number of network neurons.
func (n *Net) Store(ps []Pattern, method string) error {
	// patterns can't be nil
	if ps == nil || len(ps) == 0 {
		return fmt.Errorf("Invalid pattern supplied: %v\n", ps)
	}
	// pattern length must be the same as number of neurons
	if len(ps[0]) != len(n.neurons) {
		return fmt.Errorf("Dimension mismatch: %v\n", ps)
	}

	switch method {
	case "storkey":
		n.storeStorkey(ps)
	default:
		n.storeHebbian(ps)
	}

	return nil
}

// storeHebbian uses Hebbian learning to store patterns in Network
func (n *Net) storeHebbian(ps []Pattern) {
	// number of patterns
	pCount := float64(len(ps))
	// we only traverse higher triangular matrix because we are using Symmetric matrix
	for i := 0; i < len(n.neurons); i++ {
		for j := i + 1; j < len(n.neurons); j++ {
			for k := range ps {
				n.weights.SetSym(i, j, n.weights.At(i, j)+ps[k][i]*ps[k][j]/pCount)
			}
		}
	}
}

// storeStorkey uses Storkey learning to store patterns in Network
func (n *Net) storeStorkey(ps []Pattern) {
	// we only traverse higher triangular matrix because we are using Symmetric matrix
	pDim := float64(len(ps[0]))
	var sum float64
	for i := 0; i < len(n.neurons); i++ {
		for j := i + 1; j < len(n.neurons); j++ {
			for k := range ps {
				sum = ps[k][i] * ps[k][j]
				sum -= ps[k][i] * n.localField(ps[k], j, i)
				sum -= ps[k][j] * n.localField(ps[k], i, j)
				sum *= 1 / pDim
				n.weights.SetSym(i, j, n.weights.At(i, j)+sum)
			}
		}
	}
}

// localField calculates Storkey local field for a given pattern and returns it
func (n Net) localField(p Pattern, i, j int) float64 {
	sum := 0.0
	// calculate sum for all but i and j neuron weights
	for k := 0; k < len(n.neurons); k++ {
		if k != i && k != j {
			sum += n.weights.At(i, k) * p[k]
		}
	}

	return sum
}

// Restore tries to restore pattern from the patterns stored in the network.
// It runs Hopfield network until either a local minima (eqiters) or maxiters number of iterations has been reached.
// Restore modifies p in place so the returned pattern is closest to any of the patterns stored in the network.
// It returns error if the supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n *Net) Restore(p Pattern, maxiters, eqiters int) (Pattern, error) {
	// pattern can't be nil
	if p == nil {
		return nil, fmt.Errorf("Invalid pattern supplied: %v\n", p)
	}
	// pattern length must be the same as number of neurons
	if len(p) != len(n.neurons) {
		return nil, fmt.Errorf("Dimension mismatch: %v\n", p)
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
		neuron.state = p[i]
	}
	// we will bound the number of iterations to eqiters and maxiters
	eqiter, maxiter := 0, 0
	for maxiter < maxiters {
		// generate pseudorandom sequence
		seq := rand.Perm(len(n.neurons))
		for _, i := range seq {
			sum := 0.0
			for j := 0; j < len(n.neurons); j++ {
				// some all connections to j-th neuron
				sum += n.weights.At(i, j) * p[j]
			}
			// update pattern based on result
			switch {
			case sum >= n.bias.At(i, 0):
				p[i] = 1.0
			default:
				p[i] = -1.0
			}
			// update neuron if its state has changed
			switch n.neurons[i].ChangeState(p[i]) {
			case true:
				// if the network state changed, reset the counter
				eqiter = 0
			default:
				// if the network state hasnt changed, we are around equlibrium
				eqiter++
			}
			// if we are around equlibrium, exit
			if eqiter == eqiters {
				return p, nil
			}
		}
		maxiter++
	}

	return p, nil
}

// Energy calculates Hopfield network energy for a given pattern and returns it
// It returns error if the supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n Net) Energy(p Pattern) (float64, error) {
	// pattern can't be nil
	if p == nil {
		return 0.0, fmt.Errorf("Invalid pattern supplied: %v\n", p)
	}
	// pattern length must be the same as number of neurons
	if len(p) != len(n.neurons) {
		return 0.0, fmt.Errorf("Dimension mismatch: %v\n", p)
	}

	energy := 0.0
	// traverse the network higher triangular weights matrix
	for i := 0; i < len(n.neurons); i++ {
		for j := i + 1; j < len(n.neurons); j++ {
			energy += n.weights.At(i, j) * p[i] * p[j]
		}
	}
	bias := 0.0
	// calculate the bias additions
	for i := 0; i < len(n.neurons); i++ {
		bias += n.bias.RawVector().Data[i] * p[i]
	}

	return -(energy + bias), nil
}
