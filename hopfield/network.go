package hopfield

import (
	"fmt"
	"math"
	"math/rand"
	"strings"

	"github.com/gonum/matrix/mat64"
)

// Network is Hopfield network
type Network struct {
	// weights are network neurons weights
	weights *mat64.SymDense
	// bias are network unit direct inputs
	bias *mat64.Vector
	// method is training method
	method string
	// memorised keeps a count of memorized patterns
	memorised int
}

// NewNetwork creates new Hopfield network which is trained using the training method and returns it.
// NewNetwork returns error if either non-positive size is supplied or unsupported training method is supplied.
func NewNetwork(size int, method string) (*Network, error) {
	// can't have negative number of weights
	if size <= 0 {
		return nil, fmt.Errorf("invalid network size: %d", size)
	}
	// if unsupported method is supplied we return error
	if !strings.EqualFold("hebbian", method) && !strings.EqualFold("storkey", method) {
		return nil, fmt.Errorf("unsupported training method: %s", method)
	}
	// allocate weights and bias matrices
	weights := mat64.NewSymDense(size, nil)
	bias := mat64.NewVector(size, nil)

	return &Network{
		weights: weights,
		bias:    bias,
		method:  method,
	}, nil
}

// Weights returns network weights
func (n Network) Weights() mat64.Matrix {
	return n.weights
}

// Bias returns network bias
func (n Network) Bias() mat64.Matrix {
	return n.bias
}

// Capacity returns network capacity
func (n Network) Capacity() int {
	// c is a number of neurons
	_, c := n.weights.Dims()
	// storkey learning gives higher capacity
	if strings.EqualFold("storkey", n.method) {
		return int(math.Floor(float64(c) / (2 * math.Sqrt(math.Log(float64(c))))))
	}

	return int(math.Floor(float64(c) / (2 * math.Log(float64(c)))))
}

// Memorised returns count of memorised patterns
func (n Network) Memorised() int {
	return n.memorised
}

// Store stores supplied patterns in network.
// Store returns error if patterns is nil or if any of the patterns do not have the same dimension as number network neurons.
func (n *Network) Store(patterns []*Pattern) error {
	// patterns can't be nil
	if patterns == nil || len(patterns) == 0 {
		return fmt.Errorf("invalid patterns supplied: %v", patterns)
	}
	_, nCount := n.weights.Dims()
	// each pattern length must be the same as number of neurons
	for _, p := range patterns {
		// nil patterns are invalid
		if p == nil {
			return fmt.Errorf("invalid pattern supplied: %v", p)
		}
		// incorrect dimension
		if p.Len() != nCount {
			return fmt.Errorf("invalid pattern dimension: %d", p.Len())
		}
	}
	// store patterns in the network
	switch n.method {
	case "hebbian":
		n.storeHebbian(patterns)
	case "storkey":
		n.storeStorkey(patterns)
	}

	return nil
}

// Restore tries to restore supplied pattern from network through mode restore process and returns it.
// Mode can be either sync or async. If sync mode is requested, iters parameter is ignored.
// If async mode is requested network runs for iters iterations and returns the restored pattern.
// It returns error if invalid patterns is supplied, iters is negative or unsupported mode is supplied.
func (n *Network) Restore(p *Pattern, mode string, iters int) (*Pattern, error) {
	// pattern can't be nil
	if p == nil {
		return nil, fmt.Errorf("invalid pattern supplied: %v", p)
	}
	// pattern length must be the same as number of neurons
	_, nCount := n.weights.Dims()
	if p.Len() != nCount {
		return nil, fmt.Errorf("invalid pattern dimension: %v", p.Len())
	}
	// number of max iterations must be a positive integer
	if strings.EqualFold("async", mode) && iters <= 0 {
		return nil, fmt.Errorf("invalid number of iterations: %d", iters)
	}
	// only sync and async modes are allowed
	switch mode {
	case "sync":
		return n.restoreSync(p)
	case "async":
		return n.restoreAsync(p, iters)
	}

	return nil, fmt.Errorf("unsupported mode: %s", mode)
}

// Energy calculates Hopfield network energy for a given pattern and returns it
// It returns error if the supplied pattern is nil or if it does not have the same dimension as number of network neurons.
func (n Network) Energy(p *Pattern) (float64, error) {
	// pattern can't be nil
	if p == nil {
		return 0.0, fmt.Errorf("invalid pattern supplied: %v", p)
	}
	// pattern length must be the same as number of neurons
	_, nCount := n.weights.Dims()
	// incorrect dimension
	if p.Len() != nCount {
		return 0.0, fmt.Errorf("invalid pattern dimension: %v", p.Len())
	}
	// hopfield energy
	energy := -0.5 * mat64.Inner(p.Vec(), n.weights, p.Vec())
	energy += mat64.Dot(n.bias, p.Vec())

	return energy, nil
}

// hebbian uses Hebbian learning to generate weights matrix
func (n *Network) storeHebbian(patterns []*Pattern) {
	// pattern dimension [same as nr. of neurons]
	dim := patterns[0].Len()
	// w stores partial weights for each pattern
	w := mat64.NewSymDense(dim, nil)
	// we only traverse higher triangular matrix because we are using Symmetric matrix
	for i := 0; i < dim; i++ {
		for j := i + 1; j < dim; j++ {
			for _, p := range patterns {
				w.SetSym(i, j, w.At(i, j)+(p.At(i)*p.At(j)/float64(dim)))
			}
		}
	}
	// Add nwe weights matrix to network weights matrix
	n.weights.AddSym(n.weights, w)
}

// storkey uses Storkey learning to generate weights matrix
func (n *Network) storeStorkey(patterns []*Pattern) {
	// pattern dimension [same as nr. of neurons]
	dim := patterns[0].Len()
	// weights matrix
	w := mat64.NewSymDense(dim, nil)
	// we only traverse higher triangular matrix because we are using Symmetric matrix
	var sum float64
	for i := 0; i < dim; i++ {
		for j := i + 1; j < dim; j++ {
			for _, p := range patterns {
				sum = p.At(i) * p.At(j)
				sum -= p.At(i) * localField(w, p, j, i)
				sum -= p.At(j) * localField(w, p, i, j)
				sum *= 1 / float64(dim)
				w.SetSym(i, j, w.At(i, j)+sum)
			}
		}
	}
	// Add nwe weights matrix to network weights matrix
	n.weights.AddSym(n.weights, w)
}

// localField calculates Storkey local field for a given pattern and returns it
func localField(w *mat64.SymDense, p *Pattern, i, j int) float64 {
	sum := 0.0
	// calculate sum for all but i and j neuron weights
	for k := 0; k < p.Len(); k++ {
		if k != i && k != j {
			// TODO: Raw access turns out to be slowe than library access WTF?!
			//sum += w.RawSymmetric().Data[i*w.RawSymmetric().Stride+k] * p.At(k)
			sum += w.At(i, k) * p.At(k)
		}
	}

	return sum
}

// restoreSync restores patterns from the network synchronously
func (n *Network) restoreSync(p *Pattern) (*Pattern, error) {
	p.Vec().MulVec(n.weights, p.Vec())
	for i := 0; i < p.Len(); i++ {
		if p.At(i) >= n.bias.At(i, 0) {
			p.RawData()[i] = 1.0
		} else {
			p.RawData()[i] = -1.0
		}
	}

	return p, nil
}

// restoreAsync restores patterns from the network synchronously
func (n *Network) restoreAsync(p *Pattern, iters int) (*Pattern, error) {
	// we will bound the number of iterations to eqiters and maxiters
	var nState float64
	for iters > 0 {
		// generate pseudorandom sequence
		seq := rand.Perm(p.Len())
		for _, i := range seq {
			sum := 0.0
			for j := 0; j < p.Len(); j++ {
				// some all connections to j-th neuron
				sum += n.weights.At(i, j) * p.At(j)
			}
			// if the sum is bigger than bias
			if sum >= n.bias.At(i, 0) {
				nState = 1.0
			} else {
				nState = -1.0
			}
			if p.At(i)*nState < 0.0 {
				p.RawData()[i] = nState
			}
		}
		iters--
	}

	return p, nil
}
