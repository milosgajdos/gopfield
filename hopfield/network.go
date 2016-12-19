package hopfield

import (
	"fmt"

	"github.com/gonum/matrix/mat64"
)

// Net is Hopfield network
type Net struct {
	// units are network units
	// units store their current state
	units []float64
	// weights are network units weights
	weights *mat64.SymDense
}

// NewNet creates new Hopfield network and returns it
// If negative size is supplied NewNet returns error
func NewNet(size int) (*Net, error) {
	// can't have negative number of weights
	if size <= 0 {
		return nil, fmt.Errorf("Invalid network size: %d\n", size)
	}

	// allocate units slice
	units := make([]float64, size)
	// allocate weights matrix
	weights := mat64.NewSymDense(size, nil)

	return &Net{
		units:   units,
		weights: weights,
	}, nil
}

// Weights returns network weights
func (n Net) Weights() mat64.Symmetric {
	return n.weights
}

// Store stores new data patterns in network using Hebbian learning.
// It returns error if supplied matrix of patterns is nil.
func (n *Net) Store(data mat64.Matrix) error {
	if data == nil {
		return fmt.Errorf("Invalid data supplied: %v\n", data)
	}

	patterns, _ := data.Dims()
	// try to store all patterns
	for p := 0; p < patterns; p++ {
		// we only traverse higher triangular matrix
		for i := 0; i < len(n.units); i++ {
			for j := i + 1; j < len(n.units); j++ {
				n.weights.SetSym(i, j, n.weights.At(i, j)+(data.At(p, i)*data.At(p, j))/float64(patterns))
			}
		}
	}

	return nil
}

// Energy returns network energy
func (n Net) Energy() float64 {
	return 0.0
}
