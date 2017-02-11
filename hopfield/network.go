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

// StoreFunc stores creates symmetric matrix from supplied patterns ard returns it
type StoreFunc func([]Pattern) mat64.Symmetric

// trainMethod lists supported training methods
var store = map[string]StoreFunc{
	"hebbian": hebbian,
	"storkey": storkey,
}

// Net is Hopfield network
type Net struct {
	// network neurons
	neurons []*Neuron
	// weights are network neurons weights
	weights *mat64.SymDense
	// bias are network unit direct inputs
	bias *mat64.Vector
	// store function based on training type
	storeFunc StoreFunc
}

// NewNet creates a new Hopfield network which is trained using the requested training method and returns it.
// NewNet returns error if either non-positive size is supplied or unsupported training method is supplied.
func NewNet(size int, method string) (*Net, error) {
	// can't have negative number of weights
	if size <= 0 {
		return nil, fmt.Errorf("invalid network size: %d", size)
	}
	// if unsupported method is supplied we return error
	storeFunc, ok := store[method]
	if !ok {
		return nil, fmt.Errorf("unsupported training method: %s", method)
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
		neurons:   neurons,
		weights:   weights,
		bias:      bias,
		storeFunc: storeFunc,
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

// Bias returns network bias
func (n Net) Bias() mat64.Matrix {
	return n.bias
}

// Store stores supplied patterns in network.
// Store returns error if patterns is nil or if any of the supplied data patterns do not have the same dimension as network neurons.
func (n *Net) Store(patterns []Pattern) error {
	// patterns can't be nil
	if patterns == nil || len(patterns) == 0 {
		return fmt.Errorf("invalid patterns supplied: %v", patterns)
	}
	// each pattern length must be the same as number of neurons
	for _, p := range patterns {
		if len(p) != len(n.neurons) {
			return fmt.Errorf("pattern dimension mismatch: %v", p)
		}
	}
	// add weights matrices to the network one
	n.weights.AddSym(n.weights, n.storeFunc(patterns))

	return nil
}

// Restore tries to restore pattern from the patterns stored in the network.
// It runs Hopfield network until either a local minima (eqiters) or maxiters number of iterations has been reached.
// Restore modifies p in place so the returned pattern is closest to any of the patterns stored in the network.
// It returns error if the supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n *Net) Restore(p Pattern, maxiters, eqiters int) (Pattern, error) {
	// pattern can't be nil
	if p == nil {
		return nil, fmt.Errorf("invalid pattern supplied: %v", p)
	}
	// pattern length must be the same as number of neurons
	if len(p) != len(n.neurons) {
		return nil, fmt.Errorf("dimension mismatch: %v", p)
	}
	// number of max iterations must be a positive integer
	if maxiters <= 0 {
		return nil, fmt.Errorf("invalid number of max iterations: %d", maxiters)
	}
	// number of equlibrium iterations must be a positive integer
	if eqiters <= 0 {
		return nil, fmt.Errorf("invalid number of equilibrium iterations: %d", eqiters)
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
		return 0.0, fmt.Errorf("invalid pattern supplied: %v", p)
	}
	// pattern length must be the same as number of neurons
	if len(p) != len(n.neurons) {
		return 0.0, fmt.Errorf("dimension mismatch: %v", p)
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

// hebbian uses Hebbian learning to generate weights matrix
func hebbian(p []Pattern) mat64.Symmetric {
	// pattern dimension [same as nr. of neurons]
	dim := len(p[0])
	// weights matrix
	weights := mat64.NewSymDense(dim, nil)
	// we only traverse higher triangular matrix because we are using Symmetric matrix
	for i := 0; i < dim; i++ {
		for j := i + 1; j < dim; j++ {
			for k := range p {
				weights.SetSym(i, j, weights.At(i, j)+p[k][i]*p[k][j]/float64(dim))
			}
		}
	}

	return weights
}

// storkey uses Storkey learning to generate weights matrix
func storkey(p []Pattern) mat64.Symmetric {
	// pattern dimension [same as nr. of neurons]
	dim := len(p[0])
	// weights matrix
	weights := mat64.NewSymDense(dim, nil)

	var sum float64
	for i := 0; i < dim; i++ {
		for j := i + 1; j < dim; j++ {
			for k := range p {
				sum = p[k][i] * p[k][j]
				sum -= p[k][i] * localField(weights, p[k], j, i)
				sum -= p[k][j] * localField(weights, p[k], i, j)
				sum *= 1 / float64(dim)
				weights.SetSym(i, j, weights.At(i, j)+sum)
			}
		}
	}

	return weights
}

// localField calculates Storkey local field for a given pattern and returns it
func localField(w mat64.Symmetric, p Pattern, i, j int) float64 {
	sum := 0.0
	// calculate sum for all but i and j neuron weights
	for k := 0; k < len(p); k++ {
		if k != i && k != j {
			sum += w.At(i, k) * p[k]
		}
	}

	return sum
}
